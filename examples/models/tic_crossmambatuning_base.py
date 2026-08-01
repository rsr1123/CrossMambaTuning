import math
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from layers import RSTB
from timm.models.layers import trunc_normal_

from compressai.models.utils import conv, deconv, update_registered_buffers

SCALES_MIN = 0.11
SCALES_MAX = 256
from torch import Tensor
from .vmamba import Mamba_adapter_w_TSPG as SS2D

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

SCALES_LEVELS = 64

class SSM(nn.Module):
    def __init__(self, in_dim=128, middle_dim=64, if_first = False, adapt_factor=1,prompt_C=None):
        super().__init__()
        self.factor = adapt_factor
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        self.op = SS2D(
                d_model=in_dim,
                d_state=16,
                ssm_ratio=0.5,
                ssm_rank_ratio=0.5,
                dt_rank="auto",
                act_layer=nn.SiLU,
                if_first = if_first,
                d_conv=3,
                conv_bias=True,
                dropout=0,
                simple_init=False,
                forward_type="v2",
                shared_adapter = prompt_C
            )

    def forward(self, x,layer_id):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x) * self.gamma + x * self.gammax
        x = x.permute(0, 3, 1, 2)
        identity_x =x
        x = x.permute(0,2,3,1)
        ssm_modulate = self.op(x,layer_id).permute(0,3,1,2)
        x_tilde = identity_x + (ssm_modulate) * self.factor
        return x_tilde

class SpatialPromptAdapter(nn.Module):
    def __init__(self,
                 num_layers,
                 in_channels,
                 d_state=64,
                 embed_dim=64,
                 hidden_dim=32):
        super().__init__()

        self.d_state = d_state
        self.prefix_len = 4

        self.layer_embed = nn.Embedding(num_layers, embed_dim)

        self.global_code = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        nn.init.normal_(self.global_code, std=0.02)

        in_ch = embed_dim
        out_ch = d_state * self.prefix_len

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_ch, kernel_size=1)
        )

        self.scale = nn.Parameter(torch.full((1, 1, 1, 1), 1e-5))

    def forward(self, x_conv, layer_id):
        B = x_conv.shape[0]
        device = x_conv.device

        if not torch.is_tensor(layer_id):
            layer_id = torch.tensor(layer_id, device=device, dtype=torch.long)
        else:
            layer_id = layer_id.to(device)

        layer_vec = self.layer_embed(layer_id)
        layer_vec = layer_vec.view(1, -1, 1, 1).expand(B, -1, 1, 1)

        global_vec = self.global_code.expand(B, -1, 1, 1).to(device)

        x_in = global_vec + layer_vec

        prompt_all = self.net(x_in)

        prompt_all = self.scale * prompt_all

        B_, CD, _, _ = prompt_all.shape
        assert CD == self.d_state * self.prefix_len
        prefix = prompt_all.view(B_, 1, self.d_state, self.prefix_len)

        return prefix
class TIC_SFMA(nn.Module):

    def __init__(self, N=128, M=192, input_resolution=(256, 256), in_channel=3):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.TSPG_enc = SpatialPromptAdapter(3,64)
        self.TSPG_dec = SpatialPromptAdapter(3,64)
        self.encoder_sfmas = nn.Sequential(
            SSM(N,prompt_C= self.TSPG_enc),
            SSM(N,prompt_C= self.TSPG_enc),
            SSM(N,prompt_C= self.TSPG_enc)

        )
        self.SICA_enc = nn.Sequential(
                nn.Conv2d(N,N,kernel_size=3,stride=2,padding=1,groups=N),
                nn.Conv2d(N, 64, 1, 1, 0),
                nn.SiLU(),
                nn.Conv2d(64, N, 1, 1, 0),
                nn.SiLU()
        )
        self.SICA_enc_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1,N,1,1) * 1e-6),
            nn.Parameter(torch.ones(1,N,1,1) * 1e-6)
        ])

        self.decoder_sfmas = nn.Sequential(
            SSM(N, prompt_C = self.TSPG_dec),
            SSM(N, prompt_C = self.TSPG_dec),
            SSM(N, prompt_C = self.TSPG_dec)

        )
        self.SICA_dec = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1, groups=N),
                nn.Conv2d(N, 64, 1, 1, 0),
                nn.SiLU(),
                nn.Conv2d(64, N, 1, 1, 0),
                nn.SiLU(),
        )
        self.SICA_dec_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1,N,1,1) * 1e-6),
            nn.Parameter(torch.ones(1,N,1,1) * 1e-6)
        ])
        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint
                         )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 4, input_resolution[1] // 4),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 8, input_resolution[1] // 8),
                         depth=depths[2],
                         num_heads=num_heads[2],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                         input_resolution=(input_resolution[0] // 16, input_resolution[1] // 16),
                         depth=depths[3],
                         num_heads=num_heads[3],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 32, input_resolution[1] // 32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)

        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 64, input_resolution[1] // 64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 64, input_resolution[1] // 64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 32, input_resolution[1] // 32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size // 2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)
        self.h_s3 = deconv(N, M * 2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.g_s0 = RSTB(dim=M,
                         input_resolution=(input_resolution[0] // 16, input_resolution[1] // 16),
                         depth=depths[2],
                         num_heads=num_heads[2],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 8, input_resolution[1] // 8),
                         depth=depths[3],
                         num_heads=num_heads[3],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 4, input_resolution[1] // 4),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                         input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint)
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
        self.init_std = 0.02

        self.apply(self._init_weights)

    def g_a(self, x, x_size=None):
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)

        x = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x =self.encoder_sfmas[0](x,0)
        x1 = x
        x = self.g_a2(x)
        x = self.g_a3(x, (x_size[0]//4, x_size[1]//4))

        x = x + self.SICA_enc(x1)*self.SICA_enc_scale[0]
        x = self.encoder_sfmas[1](x,1)
        x2 = x

        x = self.g_a4(x)
        x = self.g_a5(x, (x_size[0]//8, x_size[1]//8))

        x = x + self.SICA_enc(x2)*self.SICA_enc_scale[1]
        x = self.encoder_sfmas[2](x,2)

        x = self.g_a6(x)

        x = self.g_a7(x, (x_size[0]//16, x_size[1]//16))

        return x

    def g_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        x = self.g_s1(x)
        x = self.decoder_sfmas[2](x, 2)
        x1 = x
        x = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_s3(x)

        x = x + self.SICA_dec(x1)*self.SICA_dec_scale[0]
        x = self.decoder_sfmas[1](x, 1)
        x2 = x

        x = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_s5(x)

        x = x + self.SICA_dec(x2)*self.SICA_dec_scale[1]
        x = self.decoder_sfmas[0](x, 0)
        x = self.g_s6(x, (x_size[0]//2, x_size[1]//2))

        x = self.g_s7(x)
        return x

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2] * 16, x.shape[3] * 16)
        x = self.h_a0(x)
        x = self.h_a1(x, (x_size[0] // 32, x_size[1] // 32))

        x = self.h_a2(x)
        x= self.h_a3(x, (x_size[0] // 64, x_size[1] // 64))

        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2] * 64, x.shape[3] * 64)
        x = self.h_s0(x, (x_size[0] // 64, x_size[1] // 64))

        x = self.h_s1(x)
        x = self.h_s2(x, (x_size[0] // 32, x_size[1] // 32))

        x = self.h_s3(x)

        return x
    def aux_loss(self):
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y - means_hat) + means_hat
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,"x":x,
            "y":y,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},

        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


