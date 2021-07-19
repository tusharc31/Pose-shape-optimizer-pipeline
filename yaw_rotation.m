% Uncomment these lines to test this code with the 9_42 image
% function [rot_wireframe, rot_defomationvecs] = yaw_rotation()
% offset = 90;
% ry = 3.1 + offset * pi / 180;

function [rot_wireframe, rot_defomationvecs] = yaw_rotation(trackletInfo)
offset = 90;
% offset = 180
% offset = 270
ry = trackletInfo(8) + offset * pi / 180;

% define matrix for rotation along y a-xis by ry
r = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];

% code here afterwards is copied from the rotation in initial tasks.
wireframe = (load('rotated_meanShape.txt'))';
defomationvecs = load('vectors.txt');
rot_wireframe = r*wireframe;
rot_defomationvecs = [];

for i = 1:size(defomationvecs, 1)
    temp = reshape(defomationvecs(i, :), [3, 14]);
    temp = r * temp;
    rot_defomationvecs(i, :) = reshape(temp, [1, 42]);
end

% visualizeWireframe3D(rot_wireframe)