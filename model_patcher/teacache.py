# reference: https://github.com/welltop-cn/ComfyUI-TeaCache

import logging
import torch
import comfy.model_management as mm
from comfy.ldm.wan.model import sinusoidal_embedding_1d
from unittest.mock import patch
from .utils import find_step_index_percent

SUPPORTED_MODELS_COEFFICIENTS = {
    "normal": {
        "t2v_1_3B": ([2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01], 0.08),
        "t2v_14B": ([-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404], 0.2),
        "i2v_480p_14B": ([-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01], 0.26),
        "i2v_720p_14B": ([-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683], 0.25),
    },
    "retention": {
        "t2v_1_3B": ([-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02], 0.15),
        "t2v_14B": ([-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01], 0.2),
        "i2v_480p_14B": ([2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01], 0.3),
        "i2v_720p_14B": ([8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02], 0.3),
    }
}

WEIGHT_480P = -4.353462645667605e-05

offload_device = mm.unet_offload_device()

def patch_teacache(model, model_name, mode):
    model_type = None
    if all(k in model_name for k in ("i2v", "14b", "720p")):
        model_type = "i2v_720p_14B"
    elif all(k in model_name for k in ("i2v", "14b", "480p")):
        model_type = "i2v_480p_14B"
    elif all(k in model_name for k in ("t2v", "14b")):
        model_type = "t2v_14B"
    elif all(k in model_name for k in ("t2v", "1.3b")):
        model_type = "t2v_1_3B"

    elif all(k in model_name for k in ("fun", "14b", "inp")):
        model_type = "i2v_480p_14B"

    if model_type is None:
        print("teacache model_type is None")
        return model

    new_model = model.clone()
    
    coefficients = SUPPORTED_MODELS_COEFFICIENTS[mode][model_type][0]
    rel_l1_thresh = SUPPORTED_MODELS_COEFFICIENTS[mode][model_type][1]

    print(f"patched teacache mode: {mode}, model_type: {model_type}, rel_l1_thresh: {rel_l1_thresh}")

    if 'transformer_options' not in new_model.model_options:
        new_model.model_options['transformer_options'] = {}    
            
    new_model.model_options["transformer_options"]["max_skip_steps"] = 3
    new_model.model_options["transformer_options"]["coefficients"] = coefficients
    new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
    new_model.model_options["transformer_options"]["use_ret_mode"] = "retention" in mode

    diffusion_model = new_model.get_model_object("diffusion_model")

    context = patch.multiple(
        diffusion_model,
        forward_orig=teacache_wanmodel_forward.__get__(diffusion_model, diffusion_model.__class__)
    )

    def unet_wrapper_function(model_function, kwargs):
        input = kwargs["input"]
        timestep = kwargs["timestep"]

        c = kwargs["c"]
        cond_or_uncond = kwargs["cond_or_uncond"]
        use_ret_mode = c["transformer_options"]["use_ret_mode"]

        sigmas = c["transformer_options"]["sample_sigmas"]
        current_step, current_percent = find_step_index_percent(sigmas, timestep)

        if current_step == 0:
            if (1 in cond_or_uncond) and hasattr(diffusion_model, 'teacache_state'):
                delattr(diffusion_model, 'teacache_state')

        c["transformer_options"]["current_percent"] = current_percent
        if use_ret_mode and current_percent < 0.1: # retention
            c["transformer_options"]["enable_teacache"] = False

        with context:
            return model_function(input, timestep, **c)

    new_model.set_model_unet_function_wrapper(unet_wrapper_function)

    return new_model

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result

def teacache_wanmodel_forward(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
    rel_l1_thresh = transformer_options.get("rel_l1_thresh")
    coefficients = transformer_options.get("coefficients")
    max_skip_steps = transformer_options.get("max_skip_steps")
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    use_ret_mode = transformer_options.get("use_ret_mode")
    enable_teacache = transformer_options.get("enable_teacache", True)
    

    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # context
    context = self.text_embedding(context)

    if clip_fea is not None and self.img_emb is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)


    # enable teacache
    modulated_inp = e0.to(offload_device) if use_ret_mode else e.to(offload_device)
    if not hasattr(self, 'teacache_state'):
        self.teacache_state = {
            0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None, 'skip_steps': 0},
            1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None, 'skip_steps': 0}
        }

    def update_cache_state(cache, modulated_inp):
        if cache['skip_steps'] == max_skip_steps:
            cache['should_calc'] = True
            cache['accumulated_rel_l1_distance'] = 0
            cache['skip_steps'] = 0
        elif cache['previous_modulated_input'] is not None:
            try:
                cache['accumulated_rel_l1_distance'] += poly1d(coefficients, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))
                if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                    cache['should_calc'] = False
                    cache['skip_steps'] += 1
                else:
                    cache['should_calc'] = True
                    cache['accumulated_rel_l1_distance'] = 0
                    cache['skip_steps'] = 0
            except:
                cache['should_calc'] = True
                cache['accumulated_rel_l1_distance'] = 0
                cache['skip_steps'] = 0
        cache['previous_modulated_input'] = modulated_inp
        
    b = int(len(x) / len(cond_or_uncond))

    for i, k in enumerate(cond_or_uncond):
        update_cache_state(self.teacache_state[k], modulated_inp[i*b:(i+1)*b])

    if enable_teacache:
        should_calc = False
        for k in cond_or_uncond:
            should_calc = (should_calc or self.teacache_state[k]['should_calc'])
    else:
        should_calc = True

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    if not should_calc:
        for i, k in enumerate(cond_or_uncond):
            x[i*b:(i+1)*b] += self.teacache_state[k]['previous_residual'].to(x.device)
    else:
        ori_x = x.clone()
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context)
        for i, k in enumerate(cond_or_uncond):
            self.teacache_state[k]['previous_residual'] = (x - ori_x)[i*b:(i+1)*b].to(offload_device)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x