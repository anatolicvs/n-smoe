
# from utils_n import utils_image as util
# from utils_n.utils_blindsr import degradation_bsrgan, upsample_and_clip
# import cv2
# import numpy as np
# import traceback

import torch
from torch import rand
from models.network_transformer_moe1 import BackboneDinoCfg, EncoderConfig, AutoencoderConfig, Autoencoder, BackboneResnetCfg
from models.network_transformer_moe1 import MoEConfig
# from models.network_unetmoex import ResBlock, ResBlockConfig
# from models.network_unetmoex import AttentionBlock, AttentionBlockConfig, EncoderConfig, Encoder,MoEConfig, AutoencoderConfig, Autoencoder

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    # img = util.imread_uint('utils/test.png', 1)
    # img = util.uint2single(img)
    # sf = 4
    
    # for i in range(10000):
    #     try:
    #         img_lq, img_hq = degradation_bsrgan(img, sf=sf, lq_patchsize=72)
    #         print(i)
    #     except Exception as e:
    #         print('Error:', e)
    #         traceback.print_exc()
    #         continue

    #     lq_nearest = upsample_and_clip(img_lq, sf)
    #     img_concat = np.concatenate([util.single2uint(lq_nearest), util.single2uint(img_hq)], axis=1)
    #     util.imsave(img_concat, str(i)+'.png')

    # config = ResBlockConfig(
    #     channels=64,
    #     dropout=0.1,
    #     out_channels=128,
    #     use_conv=True,
    #     dims=2,
    #     use_checkpoint=False,
    #     up=False,
    #     down=True,
    #     num_groups=32,
    #     resample_2d=True,
    # )

    # res_block = ResBlock(config)
    # print(res_block)


    # config = AttentionBlockConfig(channels=64, num_heads=8, num_head_channels=8, use_checkpoint=True, use_new_attention_order=True)
    # attention_block = AttentionBlock(cfg=config)

    # print(attention_block)

    # x = torch.rand(10, 64, 32, 32)
    # output = attention_block(x)
    # print(output.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_blocks(img_tensor, block_size, overlap):
        blocks = []
        step = block_size - overlap
        for i in range(0, img_tensor.shape[1] - block_size + 1, step):
            for j in range(0, img_tensor.shape[2] - block_size + 1, step):
                block = img_tensor[:, i:i+block_size, j:j+block_size]
                blocks.append(block)
        return torch.stack(blocks)
    
    image_tensor = torch.randn(1, 128, 128).cuda()

    phw=32
    overlap=16

    blocks = extract_blocks(image_tensor, phw, overlap)
    image_tensor = image_tensor.unsqueeze(0)

    encoder_cfg = EncoderConfig(
        embed_dim=64,
        depth=16,
        heads=16,
        num_head_channels=16,
        dim_head=64,
        mlp_dim=64,
        dropout=0.01,
        patch_size=8,
        avg_pool=False,
        scale_factor=2,
        resizer_num_layers=2,
        num_groups=64,
        activation="GELU",
        backbone_cfg = BackboneDinoCfg(
                name="dino", 
                model="dino_vitb8", 
                backbone_cfg=BackboneResnetCfg(name="resnet", model="resnet50", 
                                               num_layers=1, use_first_pool=True))
    )
    decoder_cfg = MoEConfig(
        num_mixtures=9,
        kernel=9,
        sharpening_factor=1.0
    )


    autoenocer_cfg = AutoencoderConfig(
        EncoderConfig=encoder_cfg,
        DecoderConfig=decoder_cfg,
        d_in=1,
        d_out=63,
        phw=phw,
        overlap=overlap
    )

    model = Autoencoder(
        cfg=autoenocer_cfg
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {params}")

    model = model.to(device)

    output = model(blocks, image_tensor.shape)
    print(f"Input shape: {blocks.shape} -> Output shape: {output.shape}")

    # encoder_cfg = EncoderConfig(                         
    #     model_channels=16,                 # Start with fewer channels to avoid too many parameters with small inputs
    #     num_res_blocks=4,                  # Fewer residual blocks to prevent over-parameterization
    #     attention_resolutions=[16, 8],     # Apply attention at higher resolutions only
    #     dropout=0.2,                       # Increased dropout for more regularization
    #     num_groups=8,                      # Maintain group normalization to stabilize training with small batch sizes
    #     scale_factor=4,                    # Reduced scale factor to limit downsampling given the small input size
    #     num_heads=4,                       # Fewer heads in attention mechanisms to balance model complexity
    #     num_head_channels=16,              # Fewer channels per head to reduce complexity and focus on essential features
    #     use_new_attention_order=True,
    #     use_checkpoint=True,              # Reduce memory usage since model is smaller
    #     resblock_updown=True,             # Disable resblock upsampling and downsampling
    #     channel_mult=(1, 2, 4, 8),          # Smaller channel multiplier as fewer stages of feature enhancement are needed
    #     resample_2d=True,                 # Avoid resampling in 2D to preserve spatial dimensions
    #     pool="attention"                   # Use attention pooling to focus on relevant features without spatial reduction
    # )
    
    # encoder = Encoder(encoder_cfg, d_in=3, d_out=72).cuda()
    # input_tensor = rand(1, 3, 32, 32).cuda()

    # print(encoder)

    # params = sum(p.numel() for p in encoder.parameters())
    # print(f"Total number of parameters: {params}")

    # output = encoder(input_tensor)
    # print(output.shape)


    # decoder_cfg = MoEConfig(
    #     num_mixtures=9,
    #     kernel=9,
    #     sharpening_factor=1.0
    # )

    # autoenocer_cfg = AutoencoderConfig(
    #     EncoderConfig=encoder_cfg,
    #     DecoderConfig=decoder_cfg,
    #     d_in=1,
    #     d_out=63,
    #     phw=phw,
    #     overlap=overlap
    # )

    # model = Autoencoder(
    #     cfg=autoenocer_cfg
    # )

    # print(model)

    # params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {params}")

    # model = model.to(device)

    # output = model(blocks, image_tensor.shape)
    # print(f"Input shape: {blocks.shape} -> Output shape: {output.shape}")