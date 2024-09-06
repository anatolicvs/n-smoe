gt_im = '../dataset/violeta/kiwi-matlab/gt_avg_21_512x512/V0000B0050.png';
n_im =  '../dataset/violeta/kiwi-matlab/lr_x2_256x256/V0000B0050.png';
lr_im = '../dataset/violeta/kiwi-matlab/lr_x4_128x128/V0000B0050.png';
rec_im = '../test_output/AE-MoE-v1-old_kernel-4_khw-8-opti-Adam-criter-MSE-lr-0.001-lr_min-0.0001-epochs-50-bs-16-size-64-stride-2-kp-7-sf-1-sigma-5-ntgaussian-jid-23033028/2023-10-30_22-09-29/SR-SMoE-Model_ScaleFactor-2_2023-10-30_22-11-46_V0000B0050.png';
sota_im = '../test_output/MoESR/2023-10-30_20-05-50/SR-MoESR-Model_ScaleFactor-2_2023-10-30_20-06-40_V0000B0050.png';
rec_error = '../test_output/AE-MoE-v1-old_kernel-4_khw-8-opti-Adam-criter-MSE-lr-0.001-lr_min-0.0001-epochs-50-bs-16-size-64-stride-2-kp-7-sf-1-sigma-5-ntgaussian-jid-23033028/2023-10-30_22-09-29/error_map_SR-SMoE-Model_ScaleFactor-2_2023-10-30_22-11-46_V0000B0050.png';
sota_error = '../test_output/MoESR/2023-10-30_20-05-50/error_map_SR-MoESR-Model_ScaleFactor-2_2023-10-30_20-06-40_V0000B0050.png';

display_images(gt_im, n_im, lr_im, rec_im, sota_im, rec_error, sota_error);