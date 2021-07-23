function [keypoints_aligned, confidences] = keypoints()

result_KP = (load('result_KP.txt'));
[tracklet_mat,~]=tracklets_info();
bbox_dims = [];
scaling = [];
keypoints_aligned = [];
confidences = [];

for i = 1:size(tracklet_mat, 1)
	currDims = [tracklet_mat(i, 6) - tracklet_mat(i, 4) tracklet_mat(i, 7) - tracklet_mat(i, 5)];
	bbox_dims = [bbox_dims; currDims];
    currDims = currDims./64;
    scaling = [scaling; currDims];
end

for i = 1:size(result_KP, 1)
    temp = reshape(result_KP(i, :), [3, 14]);
    ptCoords = [temp(1, :) .* scaling(i, 1) + tracklet_mat(i, 4); temp(2, :) .* scaling(i, 2) + tracklet_mat(i, 5)];
    keypoints_aligned = [keypoints_aligned; ptCoords];
    confidences = [confidences; temp(3, :)];
end