import re
import numpy as np

def ground_truth_extractor(path,seq_id):

    seq_id_int = "%04d.txt" % (seq_id)
    file_name = path+'/'+seq_id_int
    file = open(file_name,'r')
    first_line = file.readline()
    num_cols = first_line.count(" ") + 1

    file_int = open(file_name,'r')
    if num_cols == 17:
        d = np.loadtxt(file_int,
           delimiter=' ',
           dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17'),
           'formats': ('i4', 'i4', 'S4', 'i4', 'i4','float','float','float','float','float','float','float','float','float','float','float','float')})
    file.close()

    if num_cols == 18:
        d = np.loadtxt(file_int,
           delimiter=',',
           dtype={'names':  ('col1', 'col2', 'col3', 'col4', 'col5', 'col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17','col18'),
           'formats': ('i4', 'i4', 'S4', 'i4', 'i4','float','float','float','float','float','float','float','float','float','float','float','float','float')})
    file.close()
    max_images = np.amax(d['col1'])
    tracklets_list =[]
    for image_num in range(max_images+1):
        objects_list = []
        idx = np.where(d['col1'] == image_num )
        for i in idx[0]:
            objects={}
            objects["frame"] =d['col1'][i]
            objects["id"] =d['col2'][i]
            objects["type"] =d['col3'][i]
            objects["truncation"] =d['col4'][i]
            objects["occlusion"] =d['col5'][i]
            objects["alpha"] =d['col6'][i]
            objects["x1"] =d['col7'][i]
            objects["y1"] =d['col8'][i]
            objects["x2"] =d['col9'][i]
            objects["y2"] =d['col10'][i]
            objects["h"] =d['col11'][i]
            objects["w"] =d['col12'][i]
            objects["l"] =d['col13'][i]
            objects["x_"] =d['col14'][i]
            objects["y_"] =d['col15'][i]
            objects["z_"] =d['col16'][i]
            objects["ry"] =d['col17'][i]

            if(num_cols == 18):
                objects["score"] =d['col18'][i]
            objects_list.append(objects)
        tracklets_list.append(objects_list)
    return tracklets_list

 
if __name__ == "__main__":
    path = "/scratch/ragaram/keypoint_network/eval_images/devkit_tracking/python/data/tracking/label_02"
    seq_id =[ 2,10,  4,  8, 2, 9]
    frame_id= [98, 1,197,126,90,42]
    car_id=[ 1, 0, 20, 12, 1, 1]
    for i in range(len(seq_id)):      
        track_list = ground_truth_extractor(path,seq_id[i])     
        for m in range(len(track_list[frame_id[i]])):
            id = track_list[frame_id[i]][m]["id"]
            if( id == car_id[i]):
                bbox_x1 = track_list[frame_id[i]][m]["x1"]
                bbox_y1 = track_list[frame_id[i]][m]["y1"]
                bbox_x2 = track_list[frame_id[i]][m]["x2"]
                bbox_y2 = track_list[frame_id[i]][m]["y2"]
            