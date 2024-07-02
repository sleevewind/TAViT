from .others.swin_transformer import SwinTransformer
from .others.mpvit import mpvit_small as MPViT
from .tavit import TAViT
from .others.cswin import CSWin_64_12211_tiny_224
from torchvision.models import resnet50, efficientnet_b1
from .others.RepVGG import create_RepVGG_B0
from .others.biformer import biformer_small
from .others.fasternet import fasternet_s
from .others.dualvit import dualvit_s
from .others.convnext import convnext_tiny
import torch


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        # swin_finetune
        if config.MODEL.FT and model_type == 'swin':
            param = torch.load(r'./models/ckpt/swin_tiny_patch4_window7_224.pth')['model']
            param.pop("head.weight")
            param.pop("head.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'mpvit':
        model = MPViT(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/mpvit_small.pth')['model']
            param.pop("cls_head.cls.weight")
            param.pop("cls_head.cls.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'cswin':
        model = CSWin_64_12211_tiny_224(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/cswin_tiny_224.pth')['state_dict_ema']
            param.pop("head.weight")
            param.pop("head.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'res50':
        model = resnet50(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/resnet50-0676ba61.pth')
            param.pop("fc.weight")
            param.pop("fc.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'ENb1':
        model = efficientnet_b1(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/efficientnet_b1_rwightman-533bc792.pth')
            param.pop("classifier.1.weight")
            param.pop("classifier.1.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'RepVGG':
        model = create_RepVGG_B0(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/RepVGG-B0-train.pth')
            param.pop("linear.weight")
            param.pop("linear.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'biformer':
        model = biformer_small(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/biformer_small_best.pth')['model']
            param.pop("head.weight")
            param.pop("head.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'fasternet':
        model = fasternet_s(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/fasternet_s-epoch.299-val_acc1.81.2840.pth')
            param.pop("head.weight")
            param.pop("head.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'dualvit':
        model = dualvit_s(num_classes=config.MODEL.NUM_CLASSES, token_label=False)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/dualvit_s.pth')['state_dict']
            param.pop("head.weight")
            param.pop("head.bias")
            param.pop("aux_head.weight")
            param.pop("aux_head.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'convnext':
        model = convnext_tiny(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/convnext_tiny_1k_224_ema.pth')['model']
            param.pop("head.weight")
            param.pop("head.bias")
            model.load_state_dict(param, strict=False)

    elif model_type == 'TAViT':
        model = TAViT(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.TAViT.PATCH_SIZE,
            in_chans=config.MODEL.TAViT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.TAViT.EMBED_DIM,
            depths=config.MODEL.TAViT.DEPTHS,
            num_heads=config.MODEL.TAViT.NUM_HEADS,
            layer_dim=config.MODEL.TAViT.LAYER_DIM,
            mlp_ratio=config.MODEL.TAViT.MLP_RATIO,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        if config.MODEL.FT:
            param = torch.load(r'./models/ckpt/TAViT_283.pth')['model']
            param.pop("head.weight")
            param.pop("head.bias")
            model.load_state_dict(param, strict=False)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model
