function display_images(gt_im, n_im, lr_im, rec_im, sota_im, rec_error, sota_error)

% Load images
noisy_free = imread(gt_im);
noisy_image = imread(n_im);
low_res_input = imread(lr_im);
reconstruction = imread(rec_im);
sota_reconstruction = imread(sota_im);
rec_error_map = imread(rec_error);
sota_error_map = imread(sota_error);

% Create figure and set size
f = figure('Position', [50, 50, 1600, 700]);
f.Color = 'white'; % Set background color to white

% Set consistent font and size
font_name = 'Times New Roman';
font_size = 16;

% Display Ground Truth Image
subplot(2,5,1);
imshow(noisy_free, []);
title('Ground Truth Image', 'FontName', font_name, 'FontSize', font_size);
colormap gray;
axis on;
set(gca, 'XTick', []);
set(gca, 'YTick', []);
box on;

% Display Noisy Image
subplot(2,5,2);
imshow(noisy_image, []);
title('Noisy Image', 'FontName', font_name, 'FontSize', font_size);
colormap gray;
axis on;
set(gca, 'XTick', []);
set(gca, 'YTick', []);
box on;

% Display Low Resolution Input
subplot(2,5,3);
imshow(low_res_input, []);
title('Low Resolution Input', 'FontName', font_name, 'FontSize', font_size);
colormap gray;
axis on;
set(gca, 'XTick', []);
set(gca, 'YTick', []);
box on;

% Display Reconstruction
subplot(2,5,4);
imshow(reconstruction, []);
title('Reconstruction', 'FontName', font_name, 'FontSize', font_size);
colormap gray;
axis on;
set(gca, 'XTick', []);
set(gca, 'YTick', []);
box on;

% Display Reconstruction Error Map
subplot(2,5,5);
imshow(rec_error_map, []);
title('Reconstruction Error Map', 'FontName', font_name, 'FontSize', font_size);
colormap jet;
axis off;

% Display SoTA
subplot(2,5,8);
imshow(sota_reconstruction, []);
title('SoTA', 'FontName', font_name, 'FontSize', font_size);
colormap gray;
axis off;

% Display SoTA's Error Map
subplot(2,5,9);
imshow(sota_error_map, []);
title("SoTA's Error Map", 'FontName', font_name, 'FontSize', font_size);
colormap jet;
axis off;

end
