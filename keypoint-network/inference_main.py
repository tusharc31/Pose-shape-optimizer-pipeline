from __future__ import print_function, absolute_import
import os
import argparse
import matplotlib.pyplot as plt
import re
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from hour_glass import *
import torch as tf
from data_extractor import *
import cv2

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("selected device =",device)


def main(config_dict):   
    # parameter extraction 
    dataset_path = str(config_dict["label_path"])
    out_res   = int(config_dict["output_res"])
    test_input_path = str(config_dict["test_input_path"])
    model_weights    = str(config_dict["model_weights"])
    classes          = int(config_dict["classes"])
    num_stacks       = int(config_dict["stacks"])
    depth            = int(config_dict["depth"])
    num_blocks       = int(config_dict["blocks"])

    # Model creation 
    model = hg_gn(num_stacks=num_stacks,
                  depth =depth ,
                  num_blocks= num_blocks,
                  num_classes= classes
                 )
    model = torch.nn.DataParallel(model).to(device)
    
    seq_id =[ 2,10,  4,  8, 2, 9]
    frame_id= [98, 1,197,126,90,42]
    car_id=[ 1, 0, 20, 12, 1, 1]
    for i in range(len(seq_id)):      
        track_list = ground_truth_extractor(dataset_path,seq_id[i])     
        for m in range(len(track_list[frame_id[i]])):
            id = track_list[frame_id[i]][m]["id"]
            if( id == car_id[i]):
                x1 = int(track_list[frame_id[i]][m]["x1"])
                y1 = int(track_list[frame_id[i]][m]["y1"])
                x2 = int(track_list[frame_id[i]][m]["x2"])
                y2 = int(track_list[frame_id[i]][m]["y2"])
                image_name = test_input_path+'/'+str(seq_id[i])+'_'+str(frame_id[i])+'.png'
                image_name_int = str(seq_id[i])+'_'+str(frame_id[i])+'.png'
                image_rgb = cv2.imread(image_name)
                image_rgb = image_rgb[y1:y2,x1:x2,:]
                image_rgb = cv2.resize(image_rgb, (out_res *4 , out_res*4 ),interpolation = cv2.INTER_NEAREST)

                test(model,image_rgb,image_name_int,test_input_path,model_weights)

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
        vis = heatmap[ind]
        keypoint_list.append(np.array([ind[1], ind[0], vis]))

    keypoint_locations = np.stack(keypoint_list, 0).astype("float")
    keypoint_locations[:, 0] *= scale_dict[0]
    keypoint_locations[:, 1] *= scale_dict[1]

    return keypoint_locations

def keypoint_visualizer(im, kps,image_name):

         plt.imshow(im)
         for i in range(kps.shape[0]):
            if kps[i, 2] == 1:
                plt.plot(kps[i, 0], kps[i, 1], 'ro')
            else:
                plt.plot(kps[i, 0], kps[i, 1], 'bo')

         plt.savefig(image_name, dpi=300, bbox_inches='tight')
         plt.clf()



def test(model,input,image_name,output_path,model_weigths):  
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_weigths))
        model.cuda()
        print("Model loaded successfully")
    else:
        model.load_state_dict(torch.load(model_weigths, map_location=torch.device("cpu")))
        print("Model loaded successfully")

    model.eval()


    if torch.cuda.is_available():
        image_tensor = torch.from_numpy(input.transpose(2, 0, 1)).unsqueeze(0).cuda()
    else:
        image_tensor = torch.from_numpy(input.transpose(2, 0, 1)).unsqueeze(0)

    output_heatmaps = model(image_tensor.float())

    keypoint_array = heatmap_2_keypoints(output_heatmaps[-1], [4,4])

    keypoint_visualizer(input,keypoint_array,image_name)

    kp_output_name = os.path.join(output_path, image_name.split("/")[-1].split(".")[-2] + ".npy")
    print(kp_output_name)
    np.save(kp_output_name, keypoint_array)

if __name__ == '__main__':
       
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-config" ,  type=str , default="inference_config.txt", help="model configuration")
    
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