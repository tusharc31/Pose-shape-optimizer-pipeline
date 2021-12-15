from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import re
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
from torch.utils.data import  SubsetRandomSampler
from hour_glass import *
from evaluation import *
from logger import *
import jointsmseloss
import data_loader
from misc import *
from skimage.transform import resize
import torch as tf
import glob
import evaluation

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("selected device =",device)


def main(config_dict):   
    # parameter extraction 
    dataset_path = str(config_dict["data_set_path"])
    output_res   = int(config_dict["output_res"])
    is_train     = int(config_dict["is_train"])
    is_evaluate  = int(config_dict["is_evaluate"])
    is_test      = int(config_dict["is_test"])
    epochs       = int(config_dict["epochs"])
    lr           = float(config_dict["lr"])
    eval_input_path = str(config_dict["eval_input_path"])
    eval_output_path = str(config_dict["eval_output_path"])
    model_weights    = str(config_dict["model_weights"])
    batch_size       = int(config_dict["batch_size"])   
    checkpoint       = str(config_dict["checkpoint"])
    snapshot         = int(config_dict["snapshot"])
    classes          = int(config_dict["classes"])
    num_stacks       = int(config_dict["stacks"])
    depth            = int(config_dict["depth"])
    num_blocks       = int(config_dict["blocks"])


    # Logger creation 
    title = "car_keypoint_with" + '_' + "hour_glass"

    if (is_train != -1 ):
        logger = Logger(os.path.join(checkpoint, 'train_log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss','Train Acc'])
    
    if (is_evaluate != -1 ):
        logger = Logger(os.path.join(checkpoint, 'validation_log.txt'), title=title)
        logger.set_names(['Validation Loss','Validation Accuracy'])

    # Model creation 
    model = hg_gn(num_stacks=num_stacks,
                  depth =depth ,
                  num_blocks= num_blocks,
                  num_classes= classes
                 )
    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = jointsmseloss.JointsMSELoss().to(device)
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )
  
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
       
    # create data loader
    
    validation_split = 0.2
    random_seed = 42
    dataset = data_loader.keypoint_dataset(dataset_path, output_res ,train = 1,  normalize=True, jitter=True, canonical=True)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers = 2,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(dataset,
         batch_size=batch_size, 
         sampler=valid_sampler, 
         num_workers=2,
         pin_memory=True
    )

    # train
    if(is_evaluate != -1):
        validation_loss, validation_acc = validate(val_loader, model, criterion, classes,model_weights,checkpoint)
        logger.append([validation_loss ,validation_acc],is_evaluate=1)
        logger.close()

        logger.plot(['Validation Accuracy'])
        savefig(os.path.join(checkpoint, 'validation_acc_log.eps'))

        logger.plot(['Validation Loss'])
        savefig(os.path.join(checkpoint, 'validation_loss_log.eps'))
        return

    if(is_train != -1):
        schedule = [60, 90]
        best_acc = 0
        for epoch in range(0, epochs):
            lr = adjust_learning_rate(optimizer, epoch, lr, schedule, 0.1)
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
            train_loss, train_acc = train(train_loader, model, criterion, optimizer)
            #valid_loss, valid_acc = validate(val_loader, model, criterion,classes)
            logger.append([epoch + 1, lr, train_loss, train_acc],is_evaluate=0)
        
            #is_best = valid_acc > best_acc
            #best_acc = max(valid_acc, best_acc)
            #save_checkpoint({
            #    'epoch': epoch + 1,
            #    'arch': "hour glass",
            #    'state_dict': model.state_dict(),
            #    'best_acc': best_acc,
            #    'optimizer' : optimizer.state_dict(),
            #}, is_best, checkpoint=checkpoint, snapshot=snapshot)
    
        torch.save(model.state_dict(), "model.pth")
        logger.close()
        logger.plot(['Train Acc']) 
        savefig(os.path.join(checkpoint, 'train_acc_log.eps'))
        logger.plot(['Train Loss']) 
        savefig(os.path.join(checkpoint, 'train_loss_log.eps'))

    # test
    if(is_test != -1) :
    	test(model,eval_input_path,eval_output_path,model_weights)


def train(train_loader, model, criterion, optimizer):

    losses = AverageMeter()
    acces = AverageMeter()
    # switch to train mode
    model.train()

    for i, (data) in enumerate(train_loader):
        
        input = data["image"]
        target = data["heatmaps"]
         

        input, target = input.to(device), target.to(device, non_blocking=True)
        # compute output
        output = model(input)
        if type(output) == list:  # multiple output
            loss = 0
            for o in output:
                loss += criterion(o.float(), target.float())
            output = output[-1]
        else:  # single output
            loss = criterion(output.float(), target.float())
        acc = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        acces.update(acc[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg, acces.avg

def preprocess_image(image, dim = 256):
    '''
    Function to preprocess image
    '''
    h, w, c = image.shape
    image_out = resize(image, (dim, dim))[:, :, :3]

    return image_out, {"scale_y": h / (dim/4), "scale_x": w / (dim/4)}

def heatmap_2_keypoints(heatmaps, scale_dict):
    '''
    Function to obtain keypoint locations from heatmaps
    '''
    if heatmaps.is_cuda:
        heatmaps_numpy = heatmaps.detach().cpu().numpy()[0]
    else:
        heatmaps_numpy = heatmaps.detach().numpy()[0]

    keypoint_list = []

    for i in range(heatmaps_numpy.shape[0]):

        heatmap = heatmaps_numpy[i]
        ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
        #print(ind)
        #print(heatmap[ind])
        vis = heatmap[ind]
        keypoint_list.append(np.array([ind[1], ind[0], vis]))


    #keypoint_locations = spatial_soft_argmax2d(heatmaps, normalized_coordinates = False)
    keypoint_locations = np.stack(keypoint_list, 0).astype("float")
    keypoint_locations[:, 0] *= scale_dict["scale_x"]
    keypoint_locations[:, 1] *= scale_dict["scale_y"]

    return keypoint_locations

def test(model,input_path,output_path,model_weigths):
    
    images_list = glob.glob(os.path.join(input_path, "**.png"))
    images_list.sort()

    print(images_list)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_weigths))
        model.cuda()
        print("Model loaded successfully")
    else:
        model.load_state_dict(torch.load(model_weigths, map_location=torch.device("cpu")))
        print("Model loaded successfully")

    model.eval()

    for image_name in images_list:


        image_view = plt.imread(image_name)
        image_view, scale_dict = preprocess_image(image_view)

        if torch.cuda.is_available():
            image_tensor = torch.from_numpy(image_view.transpose(2, 0, 1)).unsqueeze(0).cuda()
        else:
            image_tensor = torch.from_numpy(image_view.transpose(2, 0, 1)).unsqueeze(0)

        output_heatmaps = model(image_tensor.float())

        keypoint_array = heatmap_2_keypoints(output_heatmaps[-1], scale_dict)

        kp_output_name = os.path.join(output_path, image_name.split("/")[-1].split(".")[-2] + ".npy")
        print(kp_output_name)
        np.save(kp_output_name, keypoint_array)


def validate(val_loader, model, criterion, num_classes,model_weigths,checkpoint):

    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_weigths))
        model.cuda()
        print("Model loaded successfully")
    else:
        model.load_state_dict(torch.load(model_weigths, map_location=torch.device("cpu")))
        print("Model loaded successfully")
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (data) in enumerate(val_loader):
            input = data["image"]
            target = data["heatmaps"]
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # compute output
            output = model(input)
            score_map = output[-1].cpu() if type(output) == list else output.cpu()

            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target)
                output = output[-1]
            else:  # single output
                loss = criterion(output, target)

            acc = accuracy(score_map, target.cpu())

            # take the score map (batch_size*22*64*64) and the input size batch_size * 256*256 naming of the picture should be index_bactch.png
            evaluation.create_preds(input , score_map , i,checkpoint)
            
            # measure accuracy and record loss
            
            losses.update(loss.item(), input.size(0))
            acces.update(acc[0], input.size(0))
    return losses.val_list, acces.val_list 

if __name__ == '__main__':
       
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-config" ,  type=str , default="config.txt", help="model configuration")
    
    args = parser.parse_args()
    
    config_file = args.config
    

    file_ = open(config_file,"r")
    config_dict = {}
    for line in file_:
       line   =  line.rstrip("\n")
       fields = re.split('\s+=\s+',line)
       config_dict[fields[0]] = fields[1]
    
    for key,value in config_dict.items():
       print(key , " = ",value)

    main(config_dict)



