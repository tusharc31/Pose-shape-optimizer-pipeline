function [rot_wireframe, rot_defomationvecs] = yaw_rotation(trackletInfo, wireframe, defomationvecs)

offset = 90;
ry = trackletInfo(8) + offset * pi / 180;

% define matrix for rotation along y a-xis by ry
r = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];

% code here afterwards is copied from the rotation in initial tasks.
rot_wireframe = r*wireframe;

rot_defomationvecs = [];

for i = 1:size(defomationvecs, 1)
    temp = reshape(defomationvecs(i, :), [3, 14]);
    temp = r * temp;
    rot_defomationvecs(i, :) = reshape(temp, [1, 42]);
end