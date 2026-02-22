import collections
import math
from typing import Any

import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange
from utils import sp_support

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "fast_weight", "update", "apply"])

def full_ttt_op(update_minibatch=1024, apply_only_minibatch=1024, length=10240, update_length=None):
    if update_length is None:
        update_length = length
    assert update_length % (update_minibatch + apply_only_minibatch) == 0
    config = []
    for start in range(0, update_length, update_minibatch + apply_only_minibatch):
        config.append(TTTOperator(
            start=start, end=start + update_minibatch,
            fast_weight=True, update=True, apply=False
        ))
        config.append(TTTOperator(
            start=start, end=start + update_minibatch + apply_only_minibatch,
            fast_weight=False, update=False, apply=True
        ))
    if update_length < length:
        config.append(TTTOperator(start=update_length, end=length, fast_weight=False, update=False, apply=True))
    return config


def ar_ttt_op(update_minibatch=1024, length=10240, update_length=None):
    """
    This config always apply the end of the update into the full sequence.
    """
    if update_length is None:
        update_length = length
    assert update_length % update_minibatch == 0
    config = []
    for start in range(0, update_length, update_minibatch):
        config.append(TTTOperator(
            start=start, end=start + update_minibatch,
            fast_weight=True, update=True, apply=False
        ))

    config.append(TTTOperator(start=0, end=length, fast_weight=False, update=False, apply=True))
    return config

@torch.compile
def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y

@torch.compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X

@torch.compile
def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_config: list,
    muon_update_steps: int = 0,
    elastic_lambda: float = 0.0,
    fisher_alpha: float = 0.1,
    anchor_beta: float = 0.99,
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    # Initialize anti-Fisher anchor regularization state if enabled
    use_elastic = elastic_lambda > 0.0
    if use_elastic:
        # Streaming-EMA anchors (initialized to the initial fast weights)
        w0_anchor = w0.clone()
        w1_anchor = w1.clone()
        w2_anchor = w2.clone()
        # Fisher EMA state (initialized to zeros — first chunk gets uniform regularization)
        F0 = torch.zeros_like(w0)
        F1 = torch.zeros_like(w1)
        F2 = torch.zeros_like(w2)

    output = []
    for start, end, fast_weight, update, apply in ttt_config:
        w0_now, w1_now, w2_now = w0, w1, w2

        if fast_weight:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  # bf16
            lr0i = lr0[:, start:end, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :]  # [b, l, d/1] fp32

            gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = vi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = (hidden * lr1i).transpose(-1, -2) @ vi
            w0_grad = (ki * lr0i).transpose(-1, -2) @ dgate_before_act
            w2_grad = (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul

            # all_reduce with grad to allow training as well.
            w1_grad = sp_support.sp_all_reduce(w1_grad)
            w0_grad = sp_support.sp_all_reduce(w0_grad)
            w2_grad = sp_support.sp_all_reduce(w2_grad)

            w1_grad = zeropower_via_newtonschulz5(w1_grad, muon_update_steps)
            w0_grad = zeropower_via_newtonschulz5(w0_grad, muon_update_steps)
            w2_grad = zeropower_via_newtonschulz5(w2_grad, muon_update_steps)

            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            # Anti-Fisher regularization: θ -= λ·(1 - F_norm)·(θ - θ*)
            # Important params (high F) get less regularization; unimportant params get more.
            if use_elastic:
                # Update Fisher EMA: F = α·F + (1-α)·|grad|²
                with torch.no_grad():
                    F0 = fisher_alpha * F0 + (1.0 - fisher_alpha) * w0_grad.detach().square()
                    F1 = fisher_alpha * F1 + (1.0 - fisher_alpha) * w1_grad.detach().square()
                    F2 = fisher_alpha * F2 + (1.0 - fisher_alpha) * w2_grad.detach().square()

                    # Normalize Fisher to [0, 1] per weight matrix
                    F0_norm = F0 / (F0.max() + 1e-8)
                    F1_norm = F1 / (F1.max() + 1e-8)
                    F2_norm = F2 / (F2.max() + 1e-8)

                    # Inverse importance: 1 - F_norm
                    inv_imp0 = 1.0 - F0_norm
                    inv_imp1 = 1.0 - F1_norm
                    inv_imp2 = 1.0 - F2_norm

                # Debug: log regularization statistics to file
                with torch.no_grad():
                    disp0 = (w0_now - w0_anchor).abs()
                    disp1 = (w1_now - w1_anchor).abs()
                    disp2 = (w2_now - w2_anchor).abs()
                    reg0 = (elastic_lambda * inv_imp0 * disp0)
                    reg1 = (elastic_lambda * inv_imp1 * disp1)
                    reg2 = (elastic_lambda * inv_imp2 * disp2)
                    grad0_abs = w0_grad.abs()
                    grad1_abs = w1_grad.abs()
                    grad2_abs = w2_grad.abs()

                w0_now = w0_now - elastic_lambda * inv_imp0 * (w0_now - w0_anchor)
                w1_now = w1_now - elastic_lambda * inv_imp1 * (w1_now - w1_anchor)
                w2_now = w2_now - elastic_lambda * inv_imp2 * (w2_now - w2_anchor)

            # do weight norm here
            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            if update:
                w0, w1, w2 = w0_now, w1_now, w2_now

                # Update Streaming-EMA anchor: θ* = β·θ* + (1-β)·θ
                if use_elastic:
                    w0_anchor = anchor_beta * w0_anchor + (1.0 - anchor_beta) * w0.detach()
                    w1_anchor = anchor_beta * w1_anchor + (1.0 - anchor_beta) * w1.detach()
                    w2_anchor = anchor_beta * w2_anchor + (1.0 - anchor_beta) * w2.detach()

        if apply:
            # Only calculate the output in the last repeat.
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)) @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)

    return output, w0, w1, w2

class FastWeightGluMLPMultihead(nn.Module):
    """
    On init of fast_weight:

    Let's start with the magnitude of the value.
    value_proj is initialized with uniform distribution with range [-1.0/sqrt(d), 1.0/sqrt(d)]
        x is layernormed. So during init, value is unit norm total (not per head, per head is 1.0/sqrt(num_head))
        After silu, value is around norm of 2.7 per head.  (why? seems wired)

    Then for the fast weight, assume initial lr = 0.
    Then with l2_norm of q,k, input is unit normed.
    if w0 is initialized with kaiming, relu(w0 @ q) is unit normed.
    Then w1 is initialized with kaiming, so w1 @ relu(w0 @ q) is of norm sqrt(2) per head
    Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 1,
        bias: bool = False,
        use_o_norm=True,
        base_lr=0.01,
        muon_update_steps=0,
        elastic_lambda: float = 0.0, # IF turning on, the value should be 0.05
        fisher_alpha: float = 0.5,
        anchor_beta: float = 0.8,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps
        self.elastic_lambda = elastic_lambda
        self.fisher_alpha = fisher_alpha
        self.anchor_beta = anchor_beta

        d_in = d_out = self.head_dim
        d_h = int(self.head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )  # [d_in * num_heads,  d_h]
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.lr_dim = self.num_heads
        self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)
      
        self.o_norm = torch.nn.RMSNorm(self.dim, eps=1e-5, elementwise_affine=True)
    
    def forward(self, x: torch.Tensor, vis_dict=None, shape_info=None, *args):
        """
        x: (b, l, d)
        """
        qkv = F.silu(self.to_qkv(x), inplace=True)  # Silu - Linear
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d",
            qkv=3, h=self.num_heads
        )
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_fc(x.float())  # [b, l, lr_dim]
        
        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        lr0, lr1, lr2 = rearrange(
            lr, "b l (h lrs d) -> lrs (b h) l d",
            lrs=3, h=self.num_heads, d=self.lr_dim
        )

        if "w0" in shape_info:
            assert "w1" in shape_info and "w2" in shape_info
            w0 = shape_info["w0"]
            w1 = shape_info["w1"]
            w2 = shape_info["w2"]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, shape_info["ttt_config"],
            muon_update_steps=self.muon_update_steps,
            elastic_lambda=self.elastic_lambda,
            fisher_alpha=self.fisher_alpha,
            anchor_beta=self.anchor_beta,
        )

        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
        )

        output = self.o_norm(output)
        output = self.c_proj(output)
        return output, {"w0": w0, "w1": w1, "w2": w2}
    
    def extra_repr(self) -> str:
        return (f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
                f"Muon update steps: {self.muon_update_steps}, "
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, ")
