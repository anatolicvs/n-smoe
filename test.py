# from utils_n import utils_image as util
# from utils_n.utils_blindsr import degradation_bsrgan, upsample_and_clip
# import cv2
# import numpy as np
# import traceback

# import unittest
# from torch import rand

# from models.network_transformer_moe1 import BackboneDinoCfg, EncoderConfig, AutoencoderConfig, Autoencoder, BackboneResnetCfg
# from models.network_transformer_moe1 import MoEConfig
# from models.network_unetmoex import Gaussians, MoE, ResBlock, ResBlockConfig
# from models.network_unetmoex2 import (
#     AttentionBlock,
#     AttentionBlockConfig,
#     EncoderConfig,
#     Encoder,
#     MoEConfig,
#     AutoencoderConfig,
#     Autoencoder,
# )


# class TestMoE(unittest.TestCase):

#     def setUp(self):
#         # Initialize MoE with typical configuration and input dimensions
#         cfg = MoEConfig(kernel=4, sharpening_factor=1.0)
#         self.model = MoE(cfg, d_in=3)  # Assuming RGB input for d_in

#     def test_rotation_matrix(self):
#         # Test rotation matrix calculations for basic angles
#         theta = torch.tensor(
#             [0, torch.pi / 4, torch.pi / 2, torch.pi], dtype=torch.float32
#         )
#         expected_results = [
#             torch.tensor([[1, 0], [0, 1]]),  # 0 radians
#             torch.tensor([[0.7071, -0.7071], [0.7071, 0.7071]]),  # pi/4 radians
#             torch.tensor([[0, -1], [1, 0]]),  # pi/2 radians
#             torch.tensor([[-1, 0], [0, -1]]),  # pi radians
#         ]
#         for t, expected in zip(theta, expected_results):
#             result = self.model.ang_to_rot_mat(t.unsqueeze(0))
#             self.assertTrue(
#                 torch.allclose(result, expected, atol=1e-4),
#                 f"Rotation matrix for {t.item()} radians",
#             )

#     def test_covariance_matrix_construction(self):
#         # Test covariance matrix construction correctness
#         scales = torch.tensor([[1, 2]], dtype=torch.float32)
#         theta = torch.tensor([0], dtype=torch.float32)
#         expected = torch.tensor([[[1, 0], [0, 4]]], dtype=torch.float32)
#         result = self.model.cov_mat_2d(scales, theta)
#         self.assertTrue(
#             torch.allclose(result, expected),
#             "Covariance matrix does not match expected output.",
#         )

#     def test_parameter_extraction(self):
#         # Test parameter extraction functionality
#         p = torch.randn(
#             1, 3, 20
#         )  # Simulate input parameters for a model configured with 3 channels and 4 kernels
#         gaussians = self.model.extract_params(p, self.model.kernel, self.model.alpha)
#         self.assertIsInstance(
#             gaussians, Gaussians, "Extraction does not return a Gaussians dataclass."
#         )
#         self.assertEqual(gaussians.mu.shape, (1, 3, 4, 2), "Mu shape is incorrect.")
#         self.assertEqual(
#             gaussians.cov_matrix.shape,
#             (1, 3, 4, 2, 2),
#             "Covariance matrix shape is incorrect.",
#         )
#         self.assertEqual(gaussians.w.shape, (1, 3, 4), "Weights shape is incorrect.")

#     def test_model_integration(self):
#         # Test end-to-end model functionality
#         input_tensor = torch.randn(
#             1, 3, 100, 100
#         )  # Batch size of 1, 3 channels, 100x100 spatial dimensions
#         height, width = 100, 100
#         output = self.model(height, width, input_tensor)
#         self.assertEqual(
#             output.shape, (1, 3, height, width), "Output dimensions are incorrect."
#         )


# if __name__ == "__main__":
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cuda"

# def extract_blocks(img_tensor, block_size, overlap):
#     blocks = []
#     step = block_size - overlap
#     for i in range(0, img_tensor.shape[1] - block_size + 1, step):
#         for j in range(0, img_tensor.shape[2] - block_size + 1, step):
#             block = img_tensor[:, i : i + block_size, j : j + block_size]
#             blocks.append(block)
#     return torch.stack(blocks)

# ch = 3
# w = 256
# h = 256

# image_tensor = torch.randn(ch, w, h).to(device=device)

# phw = 16
# overlap = 14

# blocks = extract_blocks(image_tensor, phw, overlap)
# image_tensor = image_tensor.unsqueeze(0)

# encoder_cfg = EncoderConfig(
#     embed_dim=32,
#     depth=4,
#     heads=4,
#     dim_head=32,
#     mlp_dim=32,
#     dropout=0.1,
#     patch_size=8,
#     scale_factor=2,
#     resizer_num_layers=2,
#     avg_pool=False,
#     num_groups=1,
#     activation="GELU",
#     backbone_cfg=BackboneDinoCfg(
#         name="dino",
#         model="dino_vits8",
#         backbone_cfg=BackboneResnetCfg(
#             name="resnet", model="resnet50", num_layers=1, use_first_pool=False
#         ),
#     ),
# )
# decoder_cfg = MoEConfig(num_mixtures=9, kernel=9, sharpening_factor=1.0)

# autoenocer_cfg = AutoencoderConfig(
#     EncoderConfig=encoder_cfg,
#     DecoderConfig=decoder_cfg,
#     d_in=1,
#     d_out=63,
#     phw=phw,
#     overlap=overlap,
# )

# model = Autoencoder(cfg=autoenocer_cfg)

# print(model)

# params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {params}")

# model = model.to(device)

# output = model(blocks, image_tensor.shape)
# print(f"Input shape: {blocks.shape} -> Output shape: {output.shape}")

# encoder_cfg = EncoderConfig(
#     model_channels=16,  # Start with fewer channels to avoid too many parameters with small inputs
#     num_res_blocks=4,  # Fewer residual blocks to prevent over-parameterization
#     attention_resolutions=[16, 8],  # Apply attention at higher resolutions only
#     dropout=0.2,  # Increased dropout for more regularization
#     num_groups=8,  # Maintain group normalization to stabilize training with small batch sizes
#     scale_factor=8,  # Reduced scale factor to limit downsampling given the small input size
#     num_heads=4,  # Fewer heads in attention mechanisms to balance model complexity
#     num_head_channels=16,  # Fewer channels per head to reduce complexity and focus on essential features
#     use_new_attention_order=True,
#     use_checkpoint=True,  # Reduce memory usage since model is smaller
#     resblock_updown=True,  # Disable resblock upsampling and downsampling
#     channel_mult=(
#         1,
#         2,
#         4,
#         8,
#     ),  # Smaller channel multiplier as fewer stages of feature enhancement are needed
#     resample_2d=True,  # Avoid resampling in 2D to preserve spatial dimensions
#     pool="attention",  # Use attention pooling to focus on relevant features without spatial reduction
# )

# encoder = Encoder(encoder_cfg, d_in=3, d_out=72).cuda()
# input_tensor = rand(1, 3, 32, 32).cuda()

# print(encoder)

# params = sum(p.numel() for p in encoder.parameters())
# print(f"Total number of parameters: {params}")

# output = encoder(input_tensor)
# print(output.shape)

# kernel = 9
# sf = 1.0
# decoder_cfg = MoEConfig(kernel=kernel, sharpening_factor=sf)

# z = 2 * kernel + 4 * kernel + kernel

# autoenocer_cfg = AutoencoderConfig(
#     EncoderConfig=encoder_cfg,
#     DecoderConfig=decoder_cfg,
#     d_in=ch,
#     d_out=z,
#     phw=phw,
#     overlap=overlap,
# )

# model = Autoencoder(cfg=autoenocer_cfg).cuda()

# print(model)

# params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {params}")

# with torch.no_grad():
#     output = model(blocks, image_tensor.shape)
#     print(f"Input shape: {blocks.shape} -> Output shape: {output.shape}")
# unittest.main()

if __name__ == "__main__":
    import torch
    from models.network_unetmoex3 import (
        EncoderConfig,
        MoEConfig,
        AutoencoderConfig,
        Autoencoder,
    )

    torch.backends.cudnn.benchmark = True

    kernel = 9
    sf = 1.0
    z = 2 * kernel + 4 * kernel + kernel

    ch = 3
    w = 128
    h = 128

    phw = 16
    overlap = 14

    def extract_blocks(img_tensor, block_size, overlap):
        blocks = []
        step = block_size - overlap
        for i in range(0, img_tensor.shape[1] - block_size + 1, step):
            for j in range(0, img_tensor.shape[2] - block_size + 1, step):
                block = img_tensor[:, i : i + block_size, j : j + block_size]
                blocks.append(block)
        return torch.stack(blocks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_cfg = EncoderConfig(
        model_channels=64,
        num_res_blocks=8,
        attention_resolutions=[16, 8, 4],
        dropout=0.2,
        channel_mult=(2, 4, 8),
        conv_resample=False,
        dims=2,
        use_checkpoint=True,
        use_fp16=False,
        num_heads=8,
        resblock_updown=False,
        num_groups=32,
        resample_2d=False,
        scale_factor=2,
        resizer_num_layers=4,
        resizer_avg_pool=False,
        activation="GELU",
        rope_theta=10000.0,
        attention_type="cross_attention",  # "attention" or "cross_attention"
    )

    decoder_cfg = MoEConfig(kernel=kernel, sharpening_factor=sf)

    autoenocer_cfg = AutoencoderConfig(
        EncoderConfig=encoder_cfg,
        DecoderConfig=decoder_cfg,
        d_in=ch,
        d_out=z,
        phw=phw,
        overlap=overlap,
    )

    model = Autoencoder(cfg=autoenocer_cfg).to(device=device)

    print(model)

    params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {params}")

    image_tensor: torch.Tensor = torch.randn(ch, w, h).to(device=device)
    blocks = extract_blocks(image_tensor, phw, overlap)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(blocks, image_tensor.shape)
        print(f"Input shape: {blocks.shape} -> Output shape: {output.shape}")

    # if __name__ == "__main__":
    #     import torch
    #     from models.network_edsr import EDSR

    #     model = EDSR(
    #         num_in_ch=1,
    #         num_out_ch=1,
    #         num_feat=256,
    #         num_block=32,
    #         upscale=2,
    #         img_range=1.0,
    #     )

    #     with torch.no_grad():
    #         img = torch.rand(1, 1, 32, 32, dtype=torch.float32)
    #         print(img.shape)
    #         print(img.max(), img.min())
    #         output = model(img)
    #         print(output.shape)

    # if __name__ == "__main__":
    # import torch
    # from models.network_discriminator import Discriminator_VGG_96
    # from models.network_discriminator import Discriminator_VGG_128

    # x = torch.rand(1, 1, 96, 96)
    # net = Discriminator_VGG_96(in_nc=1)
    # net.eval()

    # with torch.no_grad():
    #     y = net(x)

    # print(y.size())

    # x = torch.rand(1, 1, 128, 128)
    # net = Discriminator_VGG_128(in_nc=1)
    # net.eval()

    # with torch.no_grad():
    #     y = net(x)

    # print(y.size())
    # import sys
    # import numpy as np
    # import torch

    # sys.path.append("/home/ozkan/works/n-smoe/cvpr/GPEMSR/inference_code")

    # from cvpr.GPEMSR.inference_code.model.model_superhuman import UNet_PNI

    # input = np.random.random((1, 1, 1, 160, 160)).astype(np.float32)
    # x = torch.tensor(input)
    # model = UNet_PNI()
    # print(model)

    # output = model(x)
    # print(output.shape)
