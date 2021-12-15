function [weights] = kp_weights(trackletInfo, confidences)

% common = load('data').common;
kpLookup = importdata('./kpLookup_azimuth.mat');
weights = [];
wkpsWeight = 0.7;
offset=90;

for i = 1:size(confidences, 1)
	azimuth = round(trackletInfo(i, 8) * 180 / pi + offset);
	if(azimuth < 1)
		azimuth = 360 + azimuth;
	end

	kpOcc = kpLookup(round(azimuth), :);
	kpOcc = kpOcc ./ sum(kpOcc);
	temp = confidences(i, :) .* wkpsWeight + kpOcc .* (1 - wkpsWeight);
	weights = [weights; temp];
end
