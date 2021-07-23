function [] = imgPlot(trackletInfo, pts, flag)

for i = 1:size(trackletInfo, 1)
	title = sprintf('%.0f,' , trackletInfo(i, 1:3));
	title = title(1:end - 1);      % strip final comma
	figure('NumberTitle', 'off', 'Name', title);
	img = imread(strcat('left color images/', string(trackletInfo(i, 1)), '_', string(trackletInfo(i, 2)), '.png'));
	if flag
		imshow(img);
		hold on;
		plot(pts(2 * i - 1, :), pts(2 * i, :), 'linestyle', 'none', 'marker', 'o', 'MarkerFaceColor', 'm');
	else
		visualizeWireframe2D(img, pts(2 * i - 1: 2 * i, :));
    end
	pause(1);
end

