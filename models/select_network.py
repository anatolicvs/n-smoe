import functools
import torch
from torch.nn import init
import torch.nn as nn

"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']


    # ----------------------------------------
    # denoising task
    # ----------------------------------------

    # ----------------------------------------
    # DnCNN
    # ----------------------------------------
    if net_type == 'dncnn':
        from models.network_dncnn import DnCNN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],  # total number of conv layers
                   act_mode=opt_net['act_mode'])
        
    # ----------------------------------------
    # Flexible DnCNN
    # ----------------------------------------
    elif net_type == 'fdncnn':
        from models.network_dncnn import FDnCNN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],  # total number of conv layers
                   act_mode=opt_net['act_mode'])

    # ----------------------------------------
    # FFDNet
    # ----------------------------------------
    elif net_type == 'ffdnet':
        from models.network_ffdnet import FFDNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'])

    # ----------------------------------------
    # others
    # ----------------------------------------

    # ----------------------------------------
    # super-resolution task
    # ----------------------------------------

    # ----------------------------------------
    # SRMD
    # ----------------------------------------
    elif net_type == 'srmd':
        from models.network_srmd import SRMD as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # super-resolver prior of DPSR
    # ----------------------------------------
    elif net_type == 'dpsr':
        from models.network_dpsr import MSRResNet_prior as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])


    elif net_type == 'usmoe':
        from models.network_usmoe import Autoencoder as net 
        z = 2 * opt_net["kernel"] + 4 * opt_net["num_mixtures"] + opt_net["kernel"]
        netG = net(
                    in_channels=opt_net["n_channels"],
                    latent_dim=z,
                    kernel=opt_net["kernel"],
                    num_mixtures=opt_net["num_mixtures"],
                    phw=opt_net["phw"],
                    stride=opt_net["stride"],
                    layers=opt_net["layers"],
                    dropout=opt_net["dropout"],
                    model_channels=opt_net["model_channels"],
                    num_res_blocks=opt_net["num_res_blocks"],
                    attention_resolutions=opt_net["attention_resolutions"],
                    channel_mult=opt_net["channel_mult"],
                    num_heads=opt_net["num_heads"],
                    scale_factor=opt_net["scale"],
                    use_checkpoint=opt_net["use_checkpoint"],
                    pool=opt_net["pool"],
                )

    elif net_type == 'conv_smoe':
        from models.network_convsmoe import Autoencoder as net
        z = 2 * opt_net["kernel"] + 4 * opt_net["num_mixtures"] + opt_net["kernel"]
        netG = net(
                    in_channels=opt_net["n_channels"],
                    latent_dim=z,
                    kernel=opt_net["kernel"],
                    num_mixtures=opt_net["num_mixtures"],
                    phw=opt_net["phw"],
                    stride=opt_net["stride"],
                    sharpening_factor=opt_net['sharpening_factor'],
                    # depths=opt_net['depths'],
                    # dims=opt_net['dims'],
                    scale_factor=opt_net["scale"],
                    num_layers=opt_net["num_layers"],
                    avg_pool=opt_net["avg_pool"],
                )

    elif net_type == 'umoe_muller':
        from models.network_umoe import Autoencoder as net
        z = 2 * opt_net["kernel"] + 4 * opt_net["num_mixtures"] + opt_net["kernel"]
        netG = net(
            in_channels=opt_net["n_channels"],
            latent_dim=z,
            num_mixtures=opt_net["num_mixtures"],
            scale_factor=opt_net["scale"],
            stride=opt_net["stride"],
            phw=opt_net["phw"],
            dropout=opt_net["dropout"],
            model_channels=opt_net["model_channels"],
            num_res_blocks=opt_net["num_res_blocks"],
            attention_resolutions=opt_net["attention_resolutions"],
            channel_mult=opt_net["channel_mult"],
            conv_resample=opt_net["conv_resample"],
            use_fp16=opt_net["use_fp16"],
            num_head_channels=opt_net["num_head_channels"],
            num_heads=opt_net["num_heads"],
            use_checkpoint=opt_net["use_checkpoint"],
            pool=opt_net["pool"],
            num_layers=opt_net["num_layers"]
    )
        
    elif net_type == 'f_u_moe':
        from models.network_f_u_moe import Autoencoder as net
        z = 2 * opt_net["kernel"] + 4 * opt_net["num_mixtures"] + opt_net["kernel"]
        netG = net(
            in_channels=opt_net["n_channels"],
            latent_dim=z,
            num_mixtures=opt_net["num_mixtures"],
            kernel=opt_net["kernel"],
            sharpening_factor=opt_net['sharpening_factor'],
            scale_factor=opt_net["scale"],
            overlap=opt_net["overlap"],
            phw=opt_net["phw"],
            num_layers=opt_net["num_layers"],
            avg_pool=opt_net["avg_pool"],
            pre_trained=opt_net["pre_trained"]
            )
        
    elif net_type == 'transformer_moe':
        from models.network_transformer_moe import Autoencoder, EncoderConfig, MoEConfig, BackboneResnetCfg,AutoencoderConfig,BackboneDinoCfg
        z = 2 * opt_net["kernel"] + 4 * opt_net["num_mixtures"] + opt_net["kernel"]
        encoder_cfg = EncoderConfig(
            embed_dim=opt_net["embed_dim"],
            depth=opt_net["depth"],
            heads=opt_net["heads"],
            dim_head=64,
            mlp_dim=opt_net["mlp_dim"],
            dropout=opt_net["dropout"],
            patch_size=opt_net["patch_size"],
            avg_pool=opt_net["avg_pool"],
            scale_factor=opt_net["scale"],
            resizer_num_layers=opt_net["resizer_num_layers"],
            backbone_cfg = BackboneDinoCfg(
                name="dino", 
                model= opt_net["dino_model"],
                backbone_cfg=BackboneResnetCfg(name="resnet", model=opt_net["resnet_model"], 
                                               num_layers=opt_net["resnet_num_layers"], use_first_pool=opt_net["use_first_pool"]))
        )
        decoder_cfg = MoEConfig(
            num_mixtures=opt_net["num_mixtures"],
            kernel=opt_net["kernel"],
            sharpening_factor=opt_net['sharpening_factor'])
        autoenocer_cfg = AutoencoderConfig(
            EncoderConfig=encoder_cfg,
            DecoderConfig=decoder_cfg,
            d_in=opt_net["n_channels"],
            d_out=z,
            phw=opt_net["phw"],
            overlap=opt_net["overlap"])
        
        netG = Autoencoder(cfg=autoenocer_cfg)

    elif net_type == 'selfattention_transformer':
        from models.network_selfattention_transformer_moe import Autoencoder, EncoderConfig, MoEConfig, BackboneResnetCfg,AutoencoderConfig,BackboneDinoCfg,SelfAttentionTransformerCfg,ImageSelfAttentionCfg
        z = 2 * opt_net["kernel"] + 4 * opt_net["num_mixtures"] + opt_net["kernel"]
        
        encoder_cfg = EncoderConfig(
            dropout=opt_net["dropout"], 
            avg_pool=opt_net["avg_pool"], 
            scale_factor=opt_net["scale"],
            resizer_num_layers=opt_net["resizer_num_layers"], 
            backbone_cfg = BackboneDinoCfg(
                    name="dino", 
                    model=opt_net["dino_model"],
                    backbone_cfg=BackboneResnetCfg(name="resnet", model=opt_net["resnet_model"], 
                                                num_layers=opt_net["num_layers"], use_first_pool=opt_net["use_first_pool"])),

            transformer_cfg = SelfAttentionTransformerCfg(
                    self_attention=ImageSelfAttentionCfg(
                        patch_size=opt_net["patch_size"],
                        num_octaves=opt_net["num_octaves"],
                        num_layers=opt_net["num_layers"],
                        num_heads=opt_net["num_heads"],
                        d_token=opt_net["d_token"],
                        d_dot=opt_net["d_dot"],
                        d_mlp=opt_net["d_mlp"],
                    ),
                    num_layers=opt_net["num_layers"],
                    num_heads=opt_net["num_heads"],
                    d_dot=opt_net["d_dot"],
                    d_mlp=opt_net["d_mlp"],
                    downscale=opt_net["downscale"]))
        
        decoder_cfg = MoEConfig(
            num_mixtures=opt_net["num_mixtures"],
            kernel=opt_net["kernel"],
            sharpening_factor=opt_net['sharpening_factor']
        )

        autoenocer_cfg = AutoencoderConfig(
            EncoderConfig=encoder_cfg,
            DecoderConfig=decoder_cfg,
            d_in=opt_net["n_channels"],
            d_out=z,
            phw=opt_net["phw"],
            overlap=opt_net["overlap"]
        )

        netG = Autoencoder(
            cfg=autoenocer_cfg
        )

    elif net_type == 'lft_gan_v':
        from models.network_lft_v import LFT
        netG = LFT(opt_net)

    elif net_type == 'lft_gan':
        from models.network_lft import LFT
        netG = LFT(opt_net)

    elif net_type == 'lft_atnnscale':
        from models.network_lft_atnnscale import LFT
        netG = LFT(opt_net)

    # ----------------------------------------
    # modified SRResNet v0.0
    # ----------------------------------------
    elif net_type == 'msrresnet0':
        from models.network_msrresnet import MSRResNet0 as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # modified SRResNet v0.1
    # ----------------------------------------
    elif net_type == 'msrresnet1':
        from models.network_msrresnet import MSRResNet1 as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # RRDB
    # ----------------------------------------
    elif net_type == 'rrdb':  # RRDB
        from models.network_rrdb import RRDB as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # RRDBNet
    # ----------------------------------------
    elif net_type == 'rrdbnet':  # RRDBNet
        from models.network_rrdbnet import RRDBNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nf=opt_net['nf'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   sf=opt_net['scale'])

    # ----------------------------------------
    # IMDB
    # ----------------------------------------
    elif net_type == 'imdn':  # IMDB
        from models.network_imdn import IMDN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # USRNet
    # ----------------------------------------
    elif net_type == 'usrnet':  # USRNet
        from models.network_usrnet import USRNet as net
        netG = net(n_iter=opt_net['n_iter'],
                   h_nc=opt_net['h_nc'],
                   in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'],
                   downsample_mode=opt_net['downsample_mode'],
                   upsample_mode=opt_net['upsample_mode']
                   )

    # ----------------------------------------
    # Deep Residual U-Net (drunet)
    # ----------------------------------------
    elif net_type == 'drunet':
        from models.network_unet import UNetRes as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'],
                   downsample_mode=opt_net['downsample_mode'],
                   upsample_mode=opt_net['upsample_mode'],
                   bias=opt_net['bias'])

    # ----------------------------------------
    # SwinIR
    # ----------------------------------------
    elif net_type == 'swinir':
        from models.network_swinir import SwinIR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])

    # ----------------------------------------
    # VRT
    # ----------------------------------------
    elif net_type == 'vrt':
        from models.network_vrt import VRT as net
        netG = net(upscale=opt_net['upscale'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   depths=opt_net['depths'],
                   indep_reconsts=opt_net['indep_reconsts'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   spynet_path=opt_net['spynet_path'],
                   pa_frames=opt_net['pa_frames'],
                   deformable_groups=opt_net['deformable_groups'],
                   nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'])

        # ----------------------------------------
        # RVRT
        # ----------------------------------------
    elif net_type == 'rvrt':
        from models.network_rvrt import RVRT as net
        netG = net(upscale=opt_net['upscale'],
                   clip_size=opt_net['clip_size'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   num_blocks=opt_net['num_blocks'],
                   depths=opt_net['depths'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   inputconv_groups=opt_net['inputconv_groups'],
                   spynet_path=opt_net['spynet_path'],
                   deformable_groups=opt_net['deformable_groups'],
                   attention_heads=opt_net['attention_heads'],
                   attention_window=opt_net['attention_window'],
                   nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'],
                   cpu_cache_length=opt_net['cpu_cache_length'])

    # ----------------------------------------
    # others
    # ----------------------------------------
    # TODO

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                    init_type=opt_net['init_type'],
                    init_bn_type=opt_net['init_bn_type'],
                    gain=opt_net['init_gain'])


    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # discriminator_vgg_96
    # ----------------------------------------
    if net_type == 'discriminator_vgg_96':
        from models.network_discriminator import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128':
        from models.network_discriminator import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_192
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_192':
        from models.network_discriminator import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128_SN
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128_SN':
        from models.network_discriminator import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'discriminator_patchgan':
        from models.network_discriminator import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'],
                             n_layers=opt_net['n_layers'],
                             norm_type=opt_net['norm_type'])

    elif net_type == 'discriminator_unet':
        from models.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""
def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1.0,
                 std_dev=0.02, uniform_low=-0.2, uniform_high=0.2):
    """
    Initialize network weights with added flexibility for defining initialization parameters.

    Args:
        net (torch.nn.Module): Network to initialize.
        init_type (str): Type of initialization for weights ('normal', 'uniform', 'xavier_normal', 'xavier_uniform',
                         'kaiming_normal', 'kaiming_uniform', 'orthogonal').
        init_bn_type (str): Type of initialization for BatchNorm layers ('uniform', 'constant').
        gain (float): Gain factor for scale-dependent initializations.
        std_dev (float): Standard deviation for normal distribution initializations.
        uniform_low (float): Lower bound for uniform distribution initializations.
        uniform_high (float): Upper bound for uniform distribution initializations.

    Raises:
        NotImplementedError: If the initialization type or batch normalization type is not supported.
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if 'Conv' in classname or 'Linear' in classname:
            if hasattr(m, 'weight'):
               
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, std_dev)
                    m.weight.data.clamp_(-1, 1).mul_(gain)

                elif init_type == 'uniform':
                    init.uniform_(m.weight.data, uniform_low, uniform_high)
                    m.weight.data.mul_(gain)

                elif init_type == 'xavier_normal':
                    init.xavier_normal_(m.weight.data, gain=gain)

                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=gain)

                elif init_type == 'kaiming_normal':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

                elif init_type == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)

                else:
                    raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif 'BatchNorm' in classname:
            if m.affine:
                if init_bn_type == 'uniform':
                    init.uniform_(m.weight.data, uniform_low, 1.0)
                    init.constant_(m.bias.data, 0.0)
                elif init_bn_type == 'constant':
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
                else:
                    raise NotImplementedError(f'Batch normalization initialization method [{init_bn_type}] is not implemented')
                
        elif classname in ['MullerResizer']:  # Add your specific class name
            if hasattr(m, 'weights'):
                for weight in m.weights:
                    # Initialize with a small random value to avoid zero gradients
                    nn.init.uniform_(weight, -0.05, 0.05)
            if hasattr(m, 'biases'):
                for bias in m.biases:
                    nn.init.constant_(bias, 0)

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
