function [groundTruth, objsInfo] = tracklets_info(seqs, frames, ids)

label_dir = '../../training/label_02';

% seqs = [ 2,10,  4,  8, 2, 9]; %Sequences
% frames = [98, 1,197,126,90,42]; %Frames
% ids  = [ 1, 0, 20, 12, 1, 1]; %CarID's

objsInfo = [];
groundTruth = [];

cd devkit_object/matlab;

for idx = 1:size(seqs, 2)
    
    tracklet = readLabels(label_dir, seqs(idx));
    frameInfo = tracklet(frames(idx) + 1);
    info = [];
    truthInfo = [];

    for i = 1:size(frameInfo{1}, 2)
        if(frameInfo{1}(i).id == ids(idx))
            objData = frameInfo{1}(i);
            info = double([seqs(idx), objData.frame, objData.id]);
            truthInfo = info;
            
            info = [info [objData.x1, objData.y1, objData.x2, objData.y2, objData.ry]];
            truthInfo = [truthInfo objData.t];
        end
    end
    
    objsInfo = [objsInfo; info];
    groundTruth = [groundTruth; truthInfo];
end

display(objsInfo);
display(groundTruth);
cd ..;
cd ..;
