function [] = frame_alignment()

seq = [ 2,10,  4,  8, 2, 9]; % Sequences
frm = [98, 1,197,126,90,42]; % Frames
k = [721.53,0,609.55;0,721.53,172.85;0,0,1]; %Camera Intrinsics

[tracklet_info, ~] = tracklets_info();
[translation, ~] = mobili_formula();

for i = 1:size(tracklet_info, 1)
	[aligned_frame, ~] = yaw_rotation(tracklet_info(i, :));
	aligned_frame = aligned_frame + translation(i, :)';
	aligned_frame = k * aligned_frame;
	cartesian_frame = [aligned_frame(1, :) ./ aligned_frame(3, :); aligned_frame(2, :) ./ aligned_frame(3, :)];
	title = sprintf('%.0f,' , tracklet_info(i, 1:3));
	title = title(1:end - 1);      % strip final comma
	figure('NumberTitle', 'off', 'Name', title);
	img = imread(strcat('left color images/', string(tracklet_info(i, 1)), '_', string(tracklet_info(i, 2)), '.png'));
	visualizeWireframe2D(img, cartesian_frame, seq(i), frm(i));
	pause(1);
end
