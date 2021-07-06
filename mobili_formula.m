function [predicted, err] = mobili_formula(groundTruth, trackletInfo)

K = [721.53,0,609.55;0,721.53,172.85;0,0,1]; %Intrinsics
n = [0; -1; 0]; %ground plane normal
avgDims = [1.6362, 3.8600, 1.5208];
h = avgDims(3); %camera height

predicted = [];
err = [];

for i = 1:size(trackletInfo, 1)
	b = [(trackletInfo(i, 4) + trackletInfo(i, 6)) / 2; trackletInfo(i, 7); 1];
	B = (-h * inv(K) * b) ./ (n' * inv(K) * b);
	B = B + [0; -avgDims(3) / 2; avgDims(2) / 2];

	predicted = [predicted; B'];
	err = [err; abs(B' - groundTruth(i, 4:6))];
end

display(predicted);
display(err);



% 2. Depth estimation in 3D space: Read up this good paper. Make use of the formula (let's call it the Mobili Formula) under the subsection "Object 
% Localization through Ground Plane" on pages 2-3 to estimate the 3D depth to the vehicle instance.
% Let's take the camera intrinsics as K = [721.53,0,609.55;0,721.53,172.85;0,0,1]; , the average car height (given in the previous email) as the 
% camera height (since it is mounted on top of a car). Let's also assume that the ground plane is perfectly horizontal in all our test images. 
% Hopefully, you should be able to make use of the accumulated tracklet information to obtain the 'b' vector. To verify that your estimated depth 
% is fairly accurate, compare it with the ground truth translation vector from the tracklets, there is bound to be some error but it should be 
% fairly close.

% http://vision.ucsd.edu/~manu/pdf/cvpr14_groundplane.pdf
