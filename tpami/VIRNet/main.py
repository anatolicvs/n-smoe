# from networks.VIRNet import VIRAttResUNetSR

import torch
import time
from torch.profiler import profile, ProfilerActivity

# from tpami.VIRNet.networks.network_moex_v1 import (
#     Encoder,
#     EncoderConfig,
#     MoEConfig,
#     AutoencoderConfig,
#     Autoencoder,
# )

from networks.network_tformer_moex import (
    Autoencoder,
    EncoderConfig,
    MoEConfig,
    BackboneResnetCfg,
    AutoencoderConfig,
    BackboneDinoCfg,
    KernelType,
)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = 1
    ch = 3

    w = 48
    h = 48
    sf = 1

    phw = 16
    overlap = 4

    kernel = 16

    # encoder_cfg = EncoderConfig(
    #     sigma_chn=1,
    #     kernel_chn=3,
    #     dep_S=2,
    #     dep_K=3,
    #     noise_cond=True,
    #     kernel_cond=True,
    #     noise_avg=True,
    #     model_channels=16,
    #     num_res_blocks=2,
    #     attention_resolutions=[16, 8],
    #     dropout=0.0,
    #     channel_mult=(2, 4),
    #     conv_resample=False,
    #     dims=2,
    #     use_checkpoint=True,
    #     use_fp16=False,
    #     num_heads=16,
    #     num_head_channels=-1,
    #     resblock_updown=True,
    #     num_groups=16,
    #     resample_2d=False,
    #     scale_factor=4,
    #     resizer_num_layers=2,
    #     resizer_avg_pool=False,
    #     activation="LeakyReLU",  # Activation function
    #     rope_theta=10000.0,
    #     attention_type="cross_attention",  # or "cross_attention"
    # )

    # decoder_cfg = MoEConfig(
    #     kernel=kernel,
    #     sharpening_factor=1,
    # )

    # autoenocer_cfg = AutoencoderConfig(
    #     EncoderConfig=encoder_cfg,
    #     DecoderConfig=decoder_cfg,
    #     d_in=ch,
    #     d_out=z,
    #     phw=phw,
    #     overlap=overlap,
    # )

    # netG = Autoencoder(cfg=autoenocer_cfg)

    # netG.to(device)
    # netG.eval()

    encoder_cfg = EncoderConfig(
        embed_dim=16,
        depth=2,
        heads=8,
        dim_head=16,
        mlp_dim=16,
        dropout=0.1,
        patch_size=4,
        scale_factor=sf,
        resizer_num_layers=2,
        resizer_avg_pool=False,
        activation="LeakyReLU",
        backbone_cfg=BackboneDinoCfg(
            name="dino",
            model="dino_vits16",  # "dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8",
            backbone_cfg=BackboneResnetCfg(
                name="resnet",
                model="resnet18",  # "resnet18", "resnet50", "resnet101"
                num_layers=2,
                use_first_pool=True,
            ),
        ),
        kernel_chn=3,
        sigma_chn=1,
        noise_cond=True,
        kernel_cond=True,
        noise_avg=True,
    )

    decoder_cfg = MoEConfig(
        kernel=kernel,
        sharpening_factor=1,
        kernel_type=KernelType("gaussian_cauchy"),
    )

    autoencoder_cfg = AutoencoderConfig(
        EncoderConfig=encoder_cfg,
        DecoderConfig=decoder_cfg,
        d_in=ch,
        phw=phw,
        overlap=overlap,
        dep_S=5,
        dep_K=8,
    )

    netG = Autoencoder(cfg=autoencoder_cfg).to(device)
    num_params = sum(p.numel() for p in netG.parameters())

    # encoder = Encoder(cfg=encoder_cfg, phw=phw, d_in=ch, d_out=z).to(device)
    image_tensor = torch.randn(batch, ch, w, h).to(device=device)
    # # blocks = extract_blocks(image_tensor, phw, overlap)
    # image_tensor = image_tensor.unsqueeze(0)

    # num_params = sum(p.numel() for p in encoder.parameters())

    netG.eval()
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            start_time = time.time()
            y, kinfo, sigma = netG(image_tensor)
            end_time = time.time()

    print(
        f"inference time: {end_time - start_time:.4f} seconds | Number of parameters: {num_params}"
    )
    # print(f"Blocks shape: {blocks.shape}")
    print(f"Gaussians shape: {y.shape}")
    print(f"Kinfo shape: {kinfo.shape}, Sigma shape: {sigma.shape}")
    print(prof.key_averages().table(sort_by="cuda_time_total"))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = VIRAttResUNetSR(
    #     im_chn=3,  # Number of image channels (3 for RGB)
    #     sigma_chn=1,  # Number of noise channels
    #     kernel_chn=3,  # Number of kernel channels
    #     n_feat=[64, 128, 192],  # Feature sizes for different layers
    #     dep_S=5,  # Depth for SNet (noise estimator)
    #     dep_K=8,  # Depth for KNet (kernel estimator)
    #     noise_cond=True,  # Use noise conditioning
    #     kernel_cond=True,  # Use kernel conditioning
    #     n_resblocks=1,  # Number of residual blocks in RNet
    #     extra_mode="Both",  # Mode for handling extra information
    #     noise_avg=True,  # Use averaging for noise estimates
    # ).to(device)

    # model.eval()

    # batch_size = 1
    # channels = 3
    # height = 256
    # width = 256

    # random_image = torch.rand((batch_size, channels, height, width)).to(device)
    # scale_factor = 2

    # with torch.no_grad():
    #     mu, kinfo_est, sigma = model(random_image, sf=scale_factor)

    # mu_cpu = mu.squeeze(0).cpu()  # Shape: [3, H*sf, W*sf]
    # kinfo_est_cpu = kinfo_est.squeeze(0).cpu().numpy()  # Shape: [3]
    # sigma_cpu = sigma.squeeze(0).cpu().numpy()  # Shape: [1]

    # print("Estimated Kernel Info (kinfo_est):", kinfo_est_cpu)
    # print("Estimated Noise Level (sigma):", sigma_cpu)
