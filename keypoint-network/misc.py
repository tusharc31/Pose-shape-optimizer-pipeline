from __future__ import absolute_import

import os
import shutil
import torch
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def save_checkpoint(state,  is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
   
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    epoch = int(state["epoch"])
    if snapshot and epoch % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))



def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds' : preds})


def create_preds(input,score_map,i,path):
    batch_size = input.shape[0]
    scale_factor = input.shape[2]/score_map.shape[2]

    for batch in range(batch_size):
        image_name = str(path)+'/'+str(i)+"_"+str(batch)+".png"

        sorce_map_batch = score_map[batch,:,:,:]
    
        
        heatmap = sorce_map_batch.cpu().detach().numpy() 
        keypoint_list = []
        for j in range(heatmap.shape[0]):
            heatmap_numpy = heatmap[j]
            ind = np.unravel_index(np.argmax(heatmap_numpy, axis=None), heatmap_numpy.shape)
            keypoint_list.append(np.array([ind[1], ind[0]]))

        keypoint_locations =  np.array(keypoint_list).astype("float")
        keypoint_locations[:, 0] *= scale_factor
        keypoint_locations[:, 1] *= scale_factor

        img_batch = input[batch,:,:]
        im_int_1 = img_batch.permute( 1,2, 0)
        im_int = im_int_1.cpu().detach().numpy() 
      
        plt.imshow(im_int)
        for j in range(keypoint_locations.shape[0]):
            plt.plot(keypoint_locations[j, 0], keypoint_locations[j, 1], 'bo')
        
        plt.savefig(image_name, dpi=300, bbox_inches='tight')
        plt.clf()
       
        




def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

