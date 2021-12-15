function [rot_wireframe, rot_defomationvecs] = kitti_rotation()
% In original co-oridnate system given:
% positive x-axis = left of the car
% positive y-axis = front of the car
% positive z-axis = top of the car

% In the KITTI dataset co-rodinate system:
% positive x-axis = right of the car
% positive y-axis = bottom of the car
% positive z-axis = front of the car

% Hence we need to calculate the rotation matrix and 
% rotate the co-ordinates about the origin (no translation),
% to adjust it to KITTI dataset's co-orinate system.

% To do so, we will first rotate the points, 180° anti-clockwise
% about the z-axis and then 90° anti-clockwise about the x-axis.

% Rotation matrix for 180° anti-clockwise rotation about Z:
rz = [-1 0 0; 0 -1 0; 0 0 1];

% Rotation matrix for 90° anti-clockwise rotation about X:
rx = [1 0 0; 0 0 -1; 0 1 0];

% Final rotation matrix comes out to be:
r=rx*rz;

% New co-orinates will be equal to:
% Rotation Matrix * Current Wireframe

wireframe = (load('rotated_meanShape.txt'))';
defomationvecs = load('vectors.txt');

% display(r);
% display(wireframe);
rot_wireframe = r*wireframe;

rot_defomationvecs = [];

% display(size(defomationvecs, 1));

for i = 1:size(defomationvecs, 1)
    temp = reshape(defomationvecs(i, :), [3, 14]);
    temp = r * temp;
    rot_defomationvecs(i, :) = reshape(temp, [1, 42]);
end



% Required co-ordinates are:
%    -0.7157    0.6518    1.2912
%     0.8490    0.6520    1.3160
%    -0.6454    0.5493   -1.1805
%     0.7927    0.5357   -1.1545
%    -0.5063    0.1022    1.9904
%     0.6380    0.1026    2.0194
%    -0.4446   -0.0960   -1.8406
%     0.6062   -0.1014   -1.8242
%    -0.7570   -0.3626    0.5615
%     0.8792   -0.3648    0.5860
%    -0.4593   -0.8494    0.3133
%     0.5768   -0.8491    0.3254
%    -0.4171   -0.8678   -0.8987
%     0.5465   -0.8688   -0.8989
