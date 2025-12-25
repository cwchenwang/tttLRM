from einops import einsum, rearrange, repeat
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F

def get_class_by_name(name):
    parts = name.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        use_qk_norm=True,
        causal=False,
        bias=False,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)

        self.causal = causal

    def forward(self, x, vis_dict=None, *args):
        """
        x: (b, l, d)
        """
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.c_proj(x)
        return x, {}

class MLP(nn.Module):

    def __init__(self, dim, inter_multi=4, bias=False):
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x, vis_dict=None, *args):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x, vis_dict

class Block(nn.Module):
    def __init__(self, dim, bias, block_config):
        super().__init__()
        module_list = []
        self.length_dim_list = []

        for _, module_config in enumerate(block_config):
            CLASS = get_class_by_name(module_config["type"])
            module = nn.ModuleDict(
                {
                    "ln": LayerNorm(dim, bias=bias),
                    "f": CLASS(dim=dim, bias=bias, **module_config["params"]),
                }
            )
            self.length_dim_list.append(module_config.get("length_dim", "vl"))

            module_list.append(module)

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x, shape_info):
        vis_dict = None
        for module, length_dim in zip(self.module_list, self.length_dim_list):
            residual = x
            x = module["ln"](x)

            if length_dim == "l":
                b, vl, d = x.shape
                l = shape_info["num_img_tokens"]

                if shape_info.get("num_triplane_tokens", 0) > 0:
                    x, triplane_tokens = x.split([shape_info["num_input_tokens"], shape_info["num_triplane_tokens"]], dim=1)
                    triplane_tokens = triplane_tokens.reshape(b, -1, d)
                    b, vl, d = x.shape
                    triplane_tokens, vis_dict = module["f"](triplane_tokens, vis_dict, shape_info)
                x = x.reshape(b * (vl // l), l, d)
                x, vis_dict = module["f"](x, vis_dict, shape_info)
                x = x.reshape(b, vl, d)
                if shape_info.get("num_triplane_tokens", 0) > 0:
                    x = torch.cat([x, triplane_tokens], dim=1)
            else:
                x, vis_dict = module["f"](x, vis_dict, shape_info)

            x = residual + x
        return x, vis_dict
