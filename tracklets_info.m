function [trackletInfo, groundTruth] = tracklets_info()

% objsInfo is the 6x8 matrix we were required to find
% groundTruth is the actual location (in metres)

label_dir = '../../training/label_02';

% The info for left images we are testing on:
% Take as parameter in the future
seq = [ 2,10,  4,  8, 2, 9]; %Sequences
frm = [98, 1,197,126,90,42]; %frames
id  = [ 1, 0, 20, 12, 1, 1]; %CarID's

trackletInfo = [];
groundTruth = [];

% Changing path to use readLabels.m
cd devkit_object/matlab;

for idx = 1:size(seq, 2)
    
    tracklet = readLabels(label_dir, seq(idx));
    frameInfo = tracklet(frm(idx) + 1);
    
    % display("size of tracklet");
    % display(size(tracklet));
    % display("size of frame info");
    % display(size(frameInfo));
    % display("frame info");
    % display(frameInfo);
    
    info = [];
    truthInfo = [];

    for i = 1:size(frameInfo{1}, 2)
        if(frameInfo{1}(i).id == id(idx))
            objData = frameInfo{1}(i);
            % display(objData);
            info = double([seq(idx), objData.frame, objData.id]);
            truthInfo = info;
            info = [info [objData.x1, objData.y1, objData.x2, objData.y2, objData.ry]];
            truthInfo = [truthInfo objData.t];
        end
    end
    
    trackletInfo = [trackletInfo; info];
    groundTruth = [groundTruth; truthInfo];
end

% display(objsInfo);
% display(groundTruth);
cd ..;
cd ..;
