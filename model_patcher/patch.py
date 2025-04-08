# reference: https://github.com/kijai/ComfyUI-KJNodes

import torch
import torch.nn as nn
import traceback
import comfy.model_patcher
import comfy.model_management as mm
import comfy.samplers

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope
from comfy.samplers import sampling_function, CFGGuider
from .utils import find_step_index_percent
from einops import rearrange

def skip_layer_guidance(model, blocks, start_percent, end_percent):
    block_list = [int(x.strip()) for x in blocks.split(",")]
    blocks = [int(i) for i in block_list]

    m = model.clone()

    def skip(args, extra_args):
        transformer_options = extra_args.get("transformer_options", {})

        if not transformer_options:
            raise ValueError("skip_layer requires teacache to be enabled")
        
        transformer_options = extra_args["transformer_options"]

        current_percent = transformer_options["current_percent"]

        #sigmas = transformer_options["sigmas"][0].item()
        original_block = extra_args["original_block"]

        if start_percent <= current_percent <= end_percent:

            if args["img"].shape[0] == 2:
                img_uncond = args["img"][0].unsqueeze(0)

                new_cond_args = {
                    "img": args["img"][1].unsqueeze(0),
                    "txt": args["txt"][1].unsqueeze(0),
                    "vec": args["vec"][1].unsqueeze(0),
                    "pe": args["pe"][1].unsqueeze(0)
                }

                block_cond_out = original_block(new_cond_args) # calcurate only cond block in WanAttentionBlock.forward
                img_cond = block_cond_out["img"]

                out = {
                    "img": torch.cat([img_uncond, img_cond], dim=0), # just concat
                    "txt": args["txt"],
                    "vec": args["vec"],
                    "pe": args["pe"]
                }
            else:
                if transformer_options.get("cond_or_uncond") == [0]:
                    out = original_block(args)
                else:
                    out = args
        else:
            out = original_block(args)

        return out
    
    for layer in blocks:
        m.model_options = comfy.model_patcher.set_model_options_patch_replace(m.model_options, skip, "dit", "double_block", layer)
        
    return m

def optimized_scale(positive, negative):
    positive_flat = positive.reshape(positive.shape[0], -1)
    negative_flat = negative.reshape(negative.shape[0], -1)

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star.reshape([positive.shape[0]] + [1] * (positive.ndim - 1))

def patch_cfg_zero_star(model, zero_init=1):

    m = model.clone()
    
    def cfg_zero_star(args):
        guidance_scale = args['cond_scale']
        x = args['input']
        cond_p = args['cond_denoised']
        uncond_p = args['uncond_denoised']
        out = args["denoised"]

        timestep = args["sigma"]
        sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]

        current_step_index = find_step_index_percent(sigmas, timestep)[0]

        if current_step_index <= zero_init:
            print(f"\ncurrent cfg zero steps {current_step_index} <= {zero_init}\n")
            return out * 0
        
        alpha = optimized_scale(x - cond_p, x - uncond_p)
        
        return out + uncond_p * (alpha - 1.0) + guidance_scale * uncond_p * (1.0 - alpha)
    
    m.set_model_sampler_post_cfg_function(cfg_zero_star)

    return m

def modified_wan_self_attention_forward(self, x, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n * d)
        return q, k, v

    q, k, v = qkv_fn(x)

    q, k = apply_rope(q, k, freqs)

    feta_scores = get_feta_scores(q, k, self.num_frames, self.enhance_weight)

    x = optimized_attention(
        q.view(b, s, n * d),
        k.view(b, s, n * d),
        v,
        heads=self.num_heads,
    )

    x = self.o(x)

    x *= feta_scores

    return x
    
def get_feta_scores(query, key, num_frames, enhance_weight):
    img_q, img_k = query, key #torch.Size([2, 9216, 12, 128])
    
    _, ST, num_heads, head_dim = img_q.shape
    spatial_dim = ST / num_frames
    spatial_dim = int(spatial_dim)

    query_image = rearrange(
        img_q, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
    )
    key_image = rearrange(
        img_k, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
    )

    return feta_score(query_image, key_image, head_dim, num_frames, enhance_weight)

def feta_score(query_image, key_image, head_dim, num_frames, enhance_weight):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    # Zero out diagonal elements
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean for each token's attention matrix
    # Number of off-diagonal elements per matrix is n*n - n
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + enhance_weight)
    enhance_scores = enhance_scores.clamp(min=1)
    return enhance_scores

import types
class WanAttentionPatch:
    def __init__(self, num_frames, weight):
        self.num_frames = num_frames
        self.enhance_weight = weight
        
    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.num_frames = self.num_frames
            self_module.enhance_weight = self.enhance_weight
            return modified_wan_self_attention_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)


def patch_enhance_video(model, weight, latent_frames):
    if weight == 0:
        return model

    model_clone = model.clone()
    if 'transformer_options' not in model_clone.model_options:
        model_clone.model_options['transformer_options'] = {}
    model_clone.model_options["transformer_options"]["enhance_weight"] = weight
    diffusion_model = model_clone.get_model_object("diffusion_model")

    compile_settings = getattr(model.model, "compile_settings", None)
    for idx, block in enumerate(diffusion_model.blocks):
        patched_attn = WanAttentionPatch(latent_frames, weight).__get__(block.self_attn, block.__class__)
        if compile_settings is not None:
            patched_attn = torch.compile(
                patched_attn,
                mode=compile_settings["mode"],
                dynamic=compile_settings["dynamic"],
                fullgraph=compile_settings["fullgraph"],
                backend=compile_settings["backend"]
            )
        
        model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.self_attn.forward", patched_attn)
        
    return model_clone

class CFGGuider2(CFGGuider):
    
    def set_cfg(self, cfg, guidance_percent):
        self.cfg = cfg
        self.guidance_percent = guidance_percent

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        
        transformer_options = model_options.get("transformer_options", {})
        if not transformer_options:
            raise ValueError("transformer_options is empty")

        sigmas = transformer_options["sample_sigmas"]
        
        current_percent = find_step_index_percent(sigmas, timestep)[1]
        if current_percent <= self.guidance_percent:
            uncond = self.conds.get("negative", None)
            cond_scale = self.cfg
        else: # it will be increase inference speed (same to disable_cfg1_optimization = False)
            uncond = None 
            cond_scale = 1.0

        cond = self.conds.get("positive", None)

        return sampling_function(
            self.inner_model, x=x, timestep=timestep, uncond=uncond, cond=cond, cond_scale=cond_scale,
            model_options=model_options, seed=seed
        )
    

# def patch_riflex(model, latent_frames, k=6):
    
#     model = model.clone()

#     diffusion_model = model.model.diffusion_model
    
#     d = diffusion_model.dim // diffusion_model.num_heads
#     r = RifleX(
#         d, 
#         10000.0, 
#         [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
#         latent_frames,
#         k,
#     )
    
#     model.add_object_patch(f"diffusion_model.rope_embedder", r)

#     return model

# class RifleX(nn.Module):
#     def __init__(self, dim, theta, axes_dim, num_frames, k):
#         super().__init__()
#         self.dim = dim
#         self.theta = theta
#         self.axes_dim = axes_dim
#         self.num_frames = num_frames
#         self.k = k

#     def forward(self, ids):
#         n_axes = ids.shape[-1]
#         emb = torch.cat(
#             [RifleX.riflex(ids[..., i], self.axes_dim[i], self.theta, self.num_frames, self.k if i == 0 else 0) for i in range(n_axes)],
#             dim=-3,
#         )
#         return emb.unsqueeze(1)
    
#     @staticmethod
#     def riflex(pos, dim, theta, L_test, k):
#         assert dim % 2 == 0
#         if mm.is_device_mps(pos.device) or mm.is_intel_xpu() or mm.is_directml_enabled():
#             device = torch.device("cpu")
#         else:
#             device = pos.device

#         scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
#         omega = 1.0 / (theta**scale)

#         # RIFLEX modification - adjust last frequency component if L_test and k are provided
#         if k and L_test:
#             omega[k-1] = 0.9 * 2 * torch.pi / L_test

#         out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
#         out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
#         out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
#         return out.to(dtype=torch.float32, device=pos.device)