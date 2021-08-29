function [] = pose_opt_plotter(translation, rotation)

seq = [ 2,10,  4,  8, 2, 9]; % Sequences
frm = [98, 1,197,126,90,42]; % Frames
k = [721.53,0,609.55;0,721.53,172.85;0,0,1]; %Camera Intrinsics
[tracklet_info, ~] = tracklets_info();

[finalWireFrames, ~] = frame_alignment();

for i = 1:size(translation, 1)
    R = reshape(rotation(i,:), [3,3]);
    pose_opt_wireframe = (R * finalWireFrames(3*i-2:3*i,:)) + translation(i,:)';
    pose_opt_wireframe = k*pose_opt_wireframe;
    pose_opt_wireframe_img = [pose_opt_wireframe(1,:) ./ pose_opt_wireframe(3,:); pose_opt_wireframe(2,:) ./ pose_opt_wireframe(3,:)];
    title = sprintf('%.0f,' , tracklet_info(i, 1:3));
	title = title(1:end - 1);      % strip final comma
	figure('NumberTitle', 'off', 'Name', title);
	img = imread(strcat('left color images/', string(tracklet_info(i, 1)), '_', string(tracklet_info(i, 2)), '.png'));
	visualizeWireframe2D(img, pose_opt_wireframe_img, seq(i), frm(i));
	pause(2);
end

%1.3270    0.3999    1.3075;
%-0.6074    0.3733    5.2708;
%-1.3777    0.3598    1.5909;
%-4.0000    0.4854    3.8109;
%0.9253    0.3957    2.5156;
%-1.8833    0.0466   -0.8838;

%0.9997    0.0004    0.0241   -0.0004    1.0000    0.0000   -0.0241   -0.0000    0.9997;
%0.9994    0.0001   -0.0355   -0.0001    1.0000    0.0000    0.0355    0.0000    0.9994;
%0.9986   -0.0004   -0.0528    0.0004    1.0000   -0.0000    0.0528   -0.0000    0.9986;
%0.9939   -0.0001   -0.1101    0.0001    1.0000   -0.0000    0.1101   -0.0000    0.9939;
%0.9999    0.0003   -0.0171   -0.0003    1.0000    0.0000    0.0171    0.0000    0.9999;
%0.9941    0.0001   -0.1081   -0.0001    1.0000   -0.0000    0.1081    0.0000    0.9941;
