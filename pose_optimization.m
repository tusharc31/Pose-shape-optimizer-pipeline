
function [translation, rotation, poseFrames] = pose_optimization(groundTruth, trackletInfo, alignedKeypts, weights)

% groundTruth from tracklets_info.m
% trackletInfo from trcklets_info.m
% alignedKeypts from keypoints.m
% weights from kp_weights.m

poseFrames = [];
translation = [];
rotation = [];
views = 1;
pts = 14;
obs = 14;

avgDims = [1.5208 1.6362 3.86];
K = [721.53,0,609.55;0,721.53,172.85;0,0,1];
lambda = [0.250000 0.270000 0.010000 -0.080000 -0.050000];
[trans, ~] = mobili_formula();
[alignedFrames, alignedVecs] = frame_alignment();
% [alignedFrames, alignedVecs] = alignFrame(groundTruth, trackletInfo);

for i = 1:size(trans, 1)
	file = fopen('./ceres/ceres_input_singleViewPoseAdjuster.txt', 'w');
	fprintf(file, '%d %d %d\n', [views, pts, obs]);
	fprintf(file, '%f %f %f\n', trans(i, :));
	fprintf(file, '%f %f %f\n', avgDims);
	fprintf(file, '%f %f %f %f %f %f %f %f %f\n', reshape(K', [1, 9]));

	fprintf(file, '%f %f\n', alignedKeypts(2 * i - 1:2 * i, :));
	fprintf(file, '%f\n', weights(i, :)');
	fprintf(file, '%f %f %f\n', alignedFrames(3 * i - 2:3 * i, :));

	for j=1:5
		fprintf(file, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n', alignedVecs(5 * (i - 1) + j, :));
	    end

	fprintf(file, '%f %f %f %f %f\n', lambda);
	fclose(file);
    cmd = 'cd ./ceres; ./singleViewPoseAdjuster; cd $OLDPWD';
	system(cmd);
	data = importdata('./ceres/ceres_output_singleViewPoseAdjuster.txt');
	rotParams = data(1:9);
	transParams = data(10:12);
	R = reshape(rotParams, [3 3]);
	poseFrame = (R * alignedFrames(3 * i - 2:3 * i, :)) + transParams;
	translation = [translation; transParams'];
	rotation = [rotation; rotParams'];
	poseFrames = [poseFrames; poseFrame];
end
