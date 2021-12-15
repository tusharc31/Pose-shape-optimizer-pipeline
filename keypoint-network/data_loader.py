import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import glob
import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import json
sys.path.append(os.path.abspath(os.path.join('../')))
import helper_functions
import torchvision
import imageio
from   imutils import *
from scipy.stats import multivariate_normal




class keypoint_dataset(Dataset):


    def __init__(self, dataset_path, output_res, transform = None, train = 1, normalize = True, jitter = False, canonical = False):

        '''
        init class method
        '''

        print('Retrieving data')
        self.transform = transform
        self.Images, self.Masks,  self.keypoints = self.load_image_names(dataset_path, train)
        self.out_res = output_res
        print('Data retrieved')

    def __len__(self):

        '''
        length of dataset
        '''
        return len(self.Images)


    def __getitem__(self, index):

        image_rgb = cv2.imread(self.Images[index])
        image_mask = cv2.imread(self.Masks[index])
        image_rgb = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2RGB)
        image_mask = cv2.imread(self.Masks[index], cv2.IMREAD_GRAYSCALE)
        
        result = np.where(image_mask == np.amax(image_mask))
        x1 = min(result[0])
        y1 = min(result[1])
        x2 = max(result[0])
        y2 = max(result[1]) 

        image_rgb = image_rgb[x1:x2,y1:y2,:]
        h, w, c = image_rgb.shape
        image_rgb = cv2.resize(image_rgb, (self.out_res *4 , self.out_res*4 ),interpolation = cv2.INTER_NEAREST)

        keypoints_concat = np.load(self.keypoints[index])
        keypoints_concat[:,0] = keypoints_concat[:,0] - y1   # translate
        keypoints_concat[:,1] = keypoints_concat[:,1] - x1
        keypoints_concat[:,0] =  keypoints_concat[:,0] * self.out_res/w   # scale the points
        keypoints_concat[:,1] =  keypoints_concat[:,1] * self.out_res/h

        # key points used for the pose pipeline  
        keppoint_subset = np.array([keypoints_concat[3, :], 
                  keypoints_concat[0, :], 
                  keypoints_concat[2, :], 
                  keypoints_concat[1, :], 
                  keypoints_concat[19, :], 
                  keypoints_concat[16, :], 
                  keypoints_concat[18, :], 
                  keypoints_concat[17, :], 
                  keypoints_concat[21, :], 
                  keypoints_concat[20, :], 
                  keypoints_concat[11, :], 
                  keypoints_concat[8, :],
                  keypoints_concat[10, :], 
                  keypoints_concat[9, :]])

        data = {}
        data["image"] = image_rgb
        data["heatmaps"] = self.generate_target_heatmaps_batch(keppoint_subset, (self.out_res, self.out_res))
        data["keypoints"] = keppoint_subset

        transform_tensor = self.ToTensor()
        data = transform_tensor(data, self.out_res, train = 1)

        return data

    def load_image_names(self, path, train = 1):
        '''
        Load names of image, mask, and camera pose file names
        '''
        if train == 1:
            data_file_name = 'train.txt'
        elif train == 2:
            data_file_name = 'overfit.txt'
        else:
            data_file_name = 'val.txt'

        file_path = os.path.join(path, data_file_name)

        fp = open(file_path, 'r')


        mask = []
        images = []
        keypoints = []
 
        for line in fp:

            fields = line.split(' ')
            directory_name = fields[0]
            frame_number = fields[1].rstrip("\n")

            keypoint_name = frame_number+"_KeyPoints.npy"
            image_name    = frame_number+"_Color_00.png"
            mask_name     = frame_number +"_Mask_00.png"

            images.append(os.path.join(path, directory_name, image_name))
            mask.append(os.path.join(path,directory_name,mask_name))
            keypoints.append(os.path.join(path, directory_name, keypoint_name))

        images = sorted(images)
        mask = sorted(mask)
        keypoints = sorted(keypoints)
      
        return images, mask, keypoints

   
    def generate_heatmap(self, kp, image_size):
        '''
        Generate a single heatmap
        '''

        pos = np.dstack(np.mgrid[0:image_size[0]:1, 0:image_size[1]:1])
        # kp => x, y, visibility
        rv = multivariate_normal(mean=[kp[1], kp[0]], cov = [1,1])
        heatmap = rv.pdf(pos)
        #print("heatmap max ",heatmap.max())
        if heatmap.max() <= 0 :
            print("heat.max = ",heatmap.max())
            print("key points as = ",kp[1], "  ",kp[0])
            print("image size as = ",pos.shape[0])
            #exit()      
        heatmap = heatmap / heatmap.max()
        

        if kp[2] == 0:
            heatmap *= 0

        return heatmap

    def generate_target_heatmaps_batch(self, kps, image_size):
        '''
        Generate target heatmaps from keypoints
        '''
        heatmaps = []

        for i in range(kps.shape[0]):
            heatmaps.append(self.generate_heatmap(kps[i, :], image_size))

        target_heatmap = np.stack(heatmaps, axis = 0)

        return target_heatmap

    class ToTensor(object):

        def __call__(self, data_dictionary, resize_dimension = 64, train = 1):

            if train == 1:
                data_augmentation_transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.05, hue = 0.3),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
               data_augmentation_transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


            data_sample = {}
          

            data_sample["image"] = (data_augmentation_transform(data_dictionary["image"]))
            data_sample["heatmaps"] = torch.from_numpy(data_dictionary["heatmaps"])
            data_sample["keypoints"] = torch.from_numpy(data_dictionary["keypoints"])
            return data_sample

def keypoint_visualizer(im, kps):
         im = im[:,:,:,0]
         kps = kps[0,:,:]
         plt.imshow(im)
         for i in range(kps.shape[0]):
            if kps[i, 2] == 1:
                plt.plot(kps[i, 0], kps[i, 1], 'ro')
            else:
                plt.plot(kps[i, 0], kps[i, 1], 'bo')
         plt.show()



if __name__ == "__main__":


    # dataset_path = "../../data/cars_blender_prepared/"
    dataset_path = "../train_actual_small"

    data_set = keypoint_dataset(dataset_path, train = 1,  normalize=True, jitter=True, canonical=True)
    # data_set = MultiView_dataset_blender(dataset_path, train = 1, gt_depth = False, normalize=True, jitter=True, transform = MultiView_dataset_blender.ToTensor())

 
    # print(data.var(), data.max()) 
    data_loader = DataLoader(data_set, batch_size = 1, shuffle=True)

    print(data_set[0].keys())
    for batch, data in enumerate(data_loader):

        for key in data:

            print(key, " ", data[key].shape)

        print("###########################################")
        #subplot(121)
        #plt.imshow(data["heatmaps"])
        #plt.subplot(122)
        #plt.imshow(data["views"].permute(1, 2, 0))
        #plt.show()
        keypoint_visualizer(data["image"].permute(2,3,1, 0), data["keypoints"].numpy())
