import math
import torch
import torch.nn as nn
from einops import repeat

try:
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception:
    SSMODE = "mamba_ssm"
    import selective_scan_cuda

class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(
                u, delta, A, B, C, D, None, delta_bias, delta_softplus
            )
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(
                u, delta, A, B, C, D, delta_bias, delta_softplus, nrows
            )

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                None,
                delta_bias,
                dout,
                x,
                None,
                None,
                ctx.delta_softplus,
                False,
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b, c, h, w = x.shape
        ctx.shape = (b, c, h, w)
        xs = x.new_empty((b, 4, c, h * w))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(2, 3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys):
        b, c, h, w = ctx.shape
        l = h * w
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(b, 2, -1, l)
        y = ys[:, 0] + ys[:, 1].view(b, -1, w, h).transpose(2, 3).contiguous().view(b, -1, l)
        return y.view(b, -1, h, w)

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys):
        b, k, d, h, w = ys.shape
        ctx.shape = (h, w)
        ys = ys.view(b, k, d, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(b, 2, d, -1)
        y = ys[:, 0] + ys[:, 1].view(b, -1, w, h).transpose(2, 3).contiguous().view(b, d, -1)
        return y

    @staticmethod
    def backward(ctx, x):
        h, w = ctx.shape
        b, c, l = x.shape
        xs = x.new_empty((b, 4, c, l))
        xs[:, 0] = x
        xs[:, 1] = x.view(b, c, h, w).transpose(2, 3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs.view(b, 4, c, h, w)

def cross_selective_scan(
    x,
    x_proj_weight,
    x_proj_bias,
    dt_projs_weight,
    dt_projs_bias,
    A_logs,
    Ds,
    out_norm,
    nrows=-1,
    delta_softplus=True,
    to_dtype=True,
    force_fp32=True,
):
    b, d, h, w = x.shape
    _, n = A_logs.shape
    k, _, r = dt_projs_weight.shape
    l = h * w

    if nrows < 1:
        if d % 4 == 0:
            nrows = 4
        elif d % 3 == 0:
            nrows = 3
        elif d % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = CrossScan.apply(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, k, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [r, n, n], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(b, -1, l)
    dts = dts.contiguous().view(b, -1, l)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = SelectiveScan.apply(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows
    ).view(b, k, -1, h, w)
    y = CrossMerge.apply(ys)
    y = y.transpose(1, 2).contiguous()
    y = out_norm(y).view(b, h, w, -1)
    return y.to(x.dtype) if to_dtype else y

def cross_selective_scan_task_prefix_v2(
    x,
    prefix,
    x_proj_weight,
    x_proj_bias,
    dt_projs_weight,
    dt_projs_bias,
    A_logs,
    Ds,
    out_norm,
    nrows=-1,
    delta_softplus=True,
    to_dtype=True,
    force_fp32=True,
):
    if prefix is None:
        return cross_selective_scan(
            x,
            x_proj_weight,
            x_proj_bias,
            dt_projs_weight,
            dt_projs_bias,
            A_logs,
            Ds,
            out_norm,
            nrows=nrows,
            delta_softplus=delta_softplus,
            to_dtype=to_dtype,
            force_fp32=force_fp32,
        )

    b, d, h, w = x.shape
    _, n = A_logs.shape
    k, _, r = dt_projs_weight.shape
    l = h * w
    prefix_len = prefix.shape[-1]

    if nrows < 1:
        if d % 4 == 0:
            nrows = 4
        elif d % 3 == 0:
            nrows = 3
        elif d % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = CrossScan.apply(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, k, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [r, n, n], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    if prefix.shape[1] == 1 and k > 1:
        prefix = prefix.expand(-1, k, -1, -1)
    prefix = prefix.to(Bs.dtype)
    Bs = torch.cat([prefix, Bs], dim=-1)
    Cs = torch.cat([prefix, Cs], dim=-1)

    pad_x = xs.new_zeros((b, k, d, prefix_len))
    pad_dt = dts.new_zeros((b, k, d, prefix_len))
    xs = torch.cat([pad_x, xs], dim=-1).view(b, -1, l + prefix_len)
    dts = torch.cat([pad_dt, dts], dim=-1).contiguous().view(b, -1, l + prefix_len)

    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = SelectiveScan.apply(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows
    )
    ys = ys[:, :, prefix_len : l + prefix_len].view(b, k, -1, h, w)
    y = CrossMerge.apply(ys)
    y = y.transpose(1, 2).contiguous()
    y = out_norm(y).view(b, h, w, -1)
    return y.to(x.dtype) if to_dtype else y

class LIE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)

    def forward(self, x):
        return x + (self.conv1x1(x) + self.conv3x3(x) + self.conv5x5(x)) / 3

class _SS2DBase(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        forward_type="v2",
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv

        self.disable_z_act = forward_type.endswith("nozact")
        if self.disable_z_act:
            forward_type = forward_type[:-6]

        if forward_type.endswith("softmax"):
            forward_type = forward_type[:-7]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type.endswith("sigmoid"):
            forward_type = forward_type[:-7]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        self.forward_core = {"v2": self.forward_corev2}.get(forward_type, self.forward_corev2)
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act = act_layer()

        if self.d_conv > 1:
            self.conv2d = LIE(d_expand)

        self.ssm_low_rank = d_inner < d_expand
        if self.ssm_low_rank:
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        self.x_proj = [
            nn.Linear(d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(
                self.dt_rank,
                d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if simple_init:
            self.Ds = nn.Parameter(torch.ones(self.K2 * d_inner))
            self.A_logs = nn.Parameter(torch.randn(self.K2 * d_inner, self.d_state))
            self.dt_projs_weight = nn.Parameter(torch.randn(self.K, d_inner, self.dt_rank))
            self.dt_projs_bias = nn.Parameter(torch.randn(self.K, d_inner))

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _forward_scan(self, x, prompt_bias=None):
        raise NotImplementedError

    def forward_corev2(self, x, nrows=-1, channel_first=False, prompt_bias=None):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = self._forward_scan(x, prompt_bias=prompt_bias)
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def _forward_impl(self, x, prompt_bias=None):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1)
        y = self.forward_core(x, channel_first=(self.d_conv > 1), prompt_bias=prompt_bias)
        y = y * z
        return self.dropout(self.out_proj(y))

class Mamba_adapter(_SS2DBase):
    def _forward_scan(self, x, prompt_bias=None):
        return cross_selective_scan(
            x,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            getattr(self, "out_norm", None),
            nrows=1,
            delta_softplus=True,
            force_fp32=self.training,
        )

    def forward(self, x, **kwargs):
        return self._forward_impl(x)

class Mamba_adapter_w_TSPG(_SS2DBase):
    def __init__(self, *args, shared_adapter=None, **kwargs):
        self.shared_adapter = shared_adapter
        super().__init__(*args, **kwargs)

    def _forward_scan(self, x, prompt_bias=None):
        return cross_selective_scan_task_prefix_v2(
            x,
            prompt_bias,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            getattr(self, "out_norm", None),
            nrows=1,
            delta_softplus=True,
            force_fp32=self.training,
        )

    def forward(self, x, layer_id):
        prompt_bias = None
        if self.shared_adapter is not None:
            prompt_bias = self.shared_adapter(x, layer_id)
        return self._forward_impl(x, prompt_bias=prompt_bias)


