
# from utils_n import utils_image as util
# from utils_n.utils_blindsr import degradation_bsrgan, upsample_and_clip
# import cv2
# import numpy as np
# import traceback

import torch
from torch import rand
from models.network_unetmoex import ResBlock, ResBlockConfig
from models.network_unetmoex import AttentionBlock, AttentionBlockConfig, EncoderConfig, Encoder


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

    config = EncoderConfig(
        model_channels=16,
        num_res_blocks=32,
        attention_resolutions=[16, 8, 4],
        dropout=0.0 ,
        num_groups=16,
        scale_factor=2,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=True,
        use_checkpoint=True,
        resblock_updown=True,
        channel_mult=(1, 2, 4, 8, 16),
        resample_2d= False,
        pool= "adaptive",
    )

    
    encoder = Encoder(config, d_in=3, d_out=72).cuda()
    input_tensor = rand(1, 3, 256, 256).cuda()

    print(encoder)

    params = sum(p.numel() for p in encoder.parameters())
    print(f"Total number of parameters: {params}")

    output = encoder(input_tensor)
    print(output.shape)