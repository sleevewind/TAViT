# --------------------------------------------------------
# Token-Attention Transformer
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, pad=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, act_layer=None):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False)
        self.bn = norm_layer(out_ch)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)
        return x


class PatchEmbedding(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input channels.
        embed_dim (int): Number of channels in first stage.
        layer_dim (int): Number of class token channels.
    """

    def __init__(self, in_chans, embed_dim, layer_dim, act_layer=nn.Hardswish):
        super(PatchEmbedding, self).__init__()
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim // 2, 3, 2, 1, act_layer=act_layer),
            Conv2d_BN(embed_dim // 2, embed_dim, 3, 2, 1, act_layer=act_layer)
        )
        self.tokenize = nn.Sequential(
            Conv2d_BN(embed_dim, embed_dim, 3, 2, 1, act_layer=act_layer),
            Conv2d_BN(embed_dim, embed_dim, 3, 2, 1, act_layer=act_layer),
            nn.Conv2d(embed_dim, layer_dim, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.LayerNorm([layer_dim, 1, 1])
        )

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = rearrange(self.tokenize(x), 'b c h w -> b (h w) c')
        return x, cls_token


class Decoder(nn.Module):
    r""" Decoder
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        layer_dim (int): Number of class token channels.
    """

    def __init__(self, dim, num_heads, layer_dim):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.q = nn.Parameter(torch.empty(1, num_heads, 1, dim // num_heads))
        self.cls_squeeze = nn.Sequential(
            nn.Linear(dim, layer_dim),
            nn.LayerNorm(layer_dim)
        )
        trunc_normal_(self.q, std=1e-5)

    def forward(self, x):
        x = rearrange(x, 'b (nh c) h w -> b nh (h w) c', nh=self.num_heads)
        attn = self.q @ x.transpose(-2, -1)
        attn = self.softmax(attn)
        cls_token = rearrange(attn @ x, 'b h n c -> b n (h c)')
        cls_token = self.cls_squeeze(cls_token)
        return cls_token

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class Encoder(nn.Module):
    r""" Encoder

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """

    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.conv = Conv2d_BN(dim, dim, 7, 1, 3, groups=dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim)
        )

    def forward(self, x):
        x = rearrange(self.conv(x), 'b c h w -> b h w c')
        x = rearrange(self.mlp(x), 'b h w c -> b c h w')
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class TransformerBlock(nn.Module):
    r""" Block.

    Args:
        di m (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        layer_dim (int): Number of class token channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, input_resolution, num_heads, layer_dim, mlp_ratio=4., drop_path=0.,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.encoder = Encoder(dim, mlp_ratio)
        self.decoder = Decoder(dim, num_heads, layer_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x, cls_token = x
        x = x + self.drop_path(self.encoder(x))
        cls_token = torch.cat((cls_token, self.decoder(x)), dim=-1)
        return x, cls_token

    def extra_repr(self) -> str:
        return f"dim={self.dim},  num_heads={self.num_heads}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x, cls_token = x
        x = rearrange(self.conv(x), 'b c h w -> b h w c')
        x = rearrange(self.norm(x), 'b h w c -> b c h w')
        return x, cls_token


class Stage(nn.Module):
    """ A Stage of TAViT.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        layer_dim (int): Number of class token channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (Module): Patch merging module.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, layer_dim, mlp_ratio=4.,
                 drop_path=0., downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, input_resolution=input_resolution,
                             num_heads=num_heads,
                             layer_dim=layer_dim,
                             mlp_ratio=mlp_ratio,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TAViT(nn.Module):
    r""" Token-Attention Vision Transformer

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dims (tuple(int)): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        layer_dim (tuple(int)): Number of class token channels in different layers
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=120,
                 embed_dims=[96, 192, 384], depths=[3, 3, 27], num_heads=[3, 6, 12],
                 layer_dim=[48, 48, 48], mlp_ratio=2., drop_path_rate=0.1,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)

        # The dimensions of the final class token
        self.logit_dim = sum(depths[i] * layer_dim[i] for i in range(self.num_layers)) + layer_dim[0]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedding(in_chans, embed_dims[0], layer_dim[0])

        patches_resolution = [img_size // patch_size, img_size // patch_size]

        self.patches_resolution = patches_resolution
        self.patch_merge = nn.ModuleList([
            PatchMerging(dim=embed_dims[idx], out_dim=embed_dims[idx + 1])
            for idx in range(self.num_layers - 1)])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Stage(dim=int(embed_dims[i_layer]),
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          num_heads=num_heads[i_layer],
                          layer_dim=layer_dim[i_layer],
                          mlp_ratio=mlp_ratio,
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=self.patch_merge[i_layer] if (i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.head = nn.Linear(self.logit_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x, cls_token = x
        cls_token = torch.flatten(cls_token, 1)
        return cls_token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    torch.set_printoptions(precision=4, threshold=torch.inf)
    model = TAViT()
    model.eval()

    inputs = torch.rand([1, 3, 224, 224])
    # outputs = model(inputs)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    acts = ActivationCountAnalysis(model, inputs)
    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    print(f"total activations: {acts.total()}")
    print(f"total flops : {flops.total()}")
    print(f"number of parameter: {param}")
    print(f"total memory: {memory_used}")
    # print(flop_count_table(flops, max_depth=5))
