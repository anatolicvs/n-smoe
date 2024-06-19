import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils_n.utils_test import *
from data.dataset_test_lft import MultiTestSetDataLoader
from utils_n import utils_option as option
from models.network_lft import LFT
import matplotlib.pyplot as plt

def main(args):

    opt = option.parse(args.opt, is_train=True)

    experiment_dir, checkpoints_dir, log_dir = create_dir(args)

    logger = Logger(log_dir, args)


    torch.cuda.set_device(args.local_rank)
    # device = torch.device("cuda", args.local_rank)
    device = torch.device("cuda", args.local_rank)

    logger.log_string('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of test data is: %d" % length_of_tests)


    net = LFT(opt["netG"])

    ckpt_path = args.path_pre_pth
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    # start_epoch = checkpoint['epoch']
    # try:
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint['state_dict'].items():
    #         name = 'module.' + k  # add `module.`
    #         new_state_dict[name] = v
    #     # load params
    #     net.load_state_dict(new_state_dict)
    #     logger.log_string('Use pretrain model!')
    # except:
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint['state_dict'].items():
    #         new_state_dict[k] = v
    #     # load params
    #     net.load_state_dict(new_state_dict)
    #     logger.log_string('Use pretrain model!')
    net.load_state_dict(checkpoint)
    
    net = net.to(device)
    cudnn.benchmark = True


    logger.log_string('\nStart test...')
    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            psnr_epoch_test, ssim_epoch_test = test(test_loader, device, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            logger.log_string('Test on %s, psnr/ssim is %.2f/%.3f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass
        pass
    pass


def plot_images_with_original_size(LR, HR, Reconstructed, psnr_value, ssim_value):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
   
    axes[0].imshow(LR, cmap='gray')
    axes[0].set_title("Low Resolution")
    axes[0].axis('on')
    
    axes[1].imshow(HR, cmap='gray')
    axes[1].set_title("High Resolution")
    axes[1].axis('on')
    
    axes[2].imshow(Reconstructed, cmap='gray')
    axes[2].set_title(f"Reconstructed\nPSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")
    axes[2].axis('on')

    fig.tight_layout(pad=0.1)
    plt.show()

# def save_image(image, title, filename):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(image, cmap='gray')
#     ax.set_title(title)
#     ax.axis('off')
#     fig.tight_layout(pad=0.1)
#     plt.savefig(filename)
#     plt.close(fig)

# def save_images_separately(LR, HR, Reconstructed, psnr_value, ssim_value):
#     save_image(LR, "Low Resolution", "low_resolution.png")
#     save_image(HR, "High Resolution", "high_resolution.png")
#     save_image(Reconstructed, f"Reconstructed\nPSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}", "reconstructed.png")

def test(test_loader, device, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Lr_SAI_cb, Hr_SAI_cb, Lr_SAI_cr, Hr_SAI_cr) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        
        
        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*ang_res, w*angRes
        Hr_SAI_y = Hr_SAI_y.squeeze()
    
        uh, vw = Lr_SAI_y.shape
        h0, w0 = int(uh//args.ang_res), int(vw//args.ang_res)

        subLFin = LFdivide(Lr_SAI_y, args.ang_res, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFout = torch.zeros(numU, numV, args.ang_res * args.patch_size_for_test * args.scale_factor,
                               args.ang_res * args.patch_size_for_test * args.scale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u:u+1, v:v+1, :, :]
                with torch.no_grad():
                    net.eval()
                    torch.cuda.empty_cache()
                    out = net(tmp.to(device))
                    subLFout[u:u+1, v:v+1, :, :] = out.squeeze()

        Sr_4D_y = LFintegrate(subLFout, args.ang_res, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, h0 * args.scale_factor,
                              w0 * args.scale_factor)
        Sr_SAI_y = Sr_4D_y.permute(0, 2, 1, 3).reshape((h0 * args.ang_res * args.scale_factor,
                                                        w0 * args.ang_res * args.scale_factor))

        psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        
        if idx_iter == 0:
            plot_images_with_original_size(Lr_SAI_y.cpu().numpy(), Hr_SAI_y.cpu().numpy(), Sr_SAI_y.cpu().numpy(), psnr, ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


if __name__ == '__main__':
    from utils_n.option_test import args

    main(args)
