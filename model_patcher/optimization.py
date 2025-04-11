import comfy.model_patcher
import comfy.utils
import comfy.sd
import comfy.ldm.modules.attention

import os, shutil, folder_paths
inductor_cache_dir = os.path.join(folder_paths.base_path, ".inductor_cache")
triton_cache_dir = os.path.join(folder_paths.base_path, ".triton_cache", "cache")

try:
    if os.path.exists(inductor_cache_dir):
        shutil.rmtree(inductor_cache_dir)

    if os.path.exists(triton_cache_dir):
        shutil.rmtree(triton_cache_dir)

    os.makedirs(inductor_cache_dir)
    os.makedirs(triton_cache_dir)
except Exception as e:
    print(e)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

#os.environ["TRITON_CONSTRAINED_ALLOC"] = "1"
#os.environ["TRITON_AUTOTUNE_MAX_EXAMPLES"] = "5"
#os.environ["TRITON_AUTOTUNE_FAST"] = "1"
#os.environ["TORCHINDUCTOR_DISABLE"] = "1"

try:
    import torch

    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.recompile_limit = 32
    torch._dynamo.config.suppress_errors = True

    torch._inductor.config.verbose_progress = True
    torch._inductor.config.debug = False
except Exception as e:
    print("error:", e)


original_attention = comfy.ldm.modules.attention.optimized_attention
original_patch_model = comfy.model_patcher.ModelPatcher.patch_model
original_load_lora_for_models = comfy.sd.load_lora_for_models

def torch_compile_model(model, mode):
    model = model.clone()
    diffusion_model = model.get_model_object("diffusion_model")

    try:
        if hasattr(model.model, "compile_settings"):
            compile_settings = getattr(model.model, "compile_settings")
            for i, block in enumerate(diffusion_model.blocks):
                if hasattr(block, "_orig_mod"):
                    block = block._orig_mod

                compiled_block = torch.compile(
                    block,
                    fullgraph=compile_settings["fullgraph"],
                    dynamic=compile_settings["dynamic"],
                    backend=compile_settings["backend"],
                    mode=compile_settings["mode"],
                )
                model.add_object_patch(f"diffusion_model.blocks.{i}", compiled_block)

            print("compiled model:", mode)
    except Exception as e:
        print("failed to compile model error:", e)

    return model

def patch_sage_attention(mode):
    comfy.ldm.modules.attention.optimized_attention = original_attention
    comfy.ldm.wan.model.optimized_attention = original_attention

    if mode != "disabled":
        try:
            from sageattention import sageattn, sageattn_qk_int8_pv_fp16_triton
            
            def set_sage_func():
                if mode == "auto":
                    def f(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    return f
                elif mode == "triton":
                    def f(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    return f
                    
            cached_sage_method = set_sage_func()

            @torch.compiler.disable()
            def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
                if skip_reshape:
                    b, _, _, dim_head = q.shape
                    tensor_layout="HND"
                else:
                    b, _, dim_head = q.shape
                    dim_head //= heads
                    q, k, v = map(
                        lambda t: t.view(b, -1, heads, dim_head),
                        (q, k, v),
                    )
                    tensor_layout="NHD"
                if mask is not None:
                    # add a batch dimension if there isn't already one
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    # add a heads dimension if there isn't already one
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)

                out = cached_sage_method(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)

                if tensor_layout == "HND":
                    if not skip_output_reshape:
                        out = (
                            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                        )
                else:
                    if skip_output_reshape:
                        out = out.transpose(1, 2)
                    else:
                        out = out.reshape(b, -1, heads * dim_head)
                return out
            
            comfy.ldm.modules.attention.optimized_attention = attention_sage
            comfy.ldm.wan.model.optimized_attention = attention_sage

            print("patched sage_attention:", mode)

        except Exception as e:
            print("failed to patch sage_attention error:", e)

def patch_model_order(weight_first=True):
    # model.object_patches_backup.clear()
    # print("cleared object_patches_backup keys")

    if weight_first:
        print("patched patch model order (Weight First)")
        comfy.model_patcher.ModelPatcher.patch_model = patched_patch_model
        comfy.sd.load_lora_for_models = patched_load_lora_for_models
    else:
        comfy.model_patcher.ModelPatcher.patch_model = original_patch_model
        comfy.sd.load_lora_for_models = original_load_lora_for_models

# try:
#     pass
#     import logging
#     import torch._logging
#     torch._logging.set_logs(dynamo=logging.WARNING)
# except:
#     pass

# import traceback
# def print_filtered_stack(filter_keyword="custom_nodes"):
#     stack = traceback.extract_stack()
#     filtered_stack = [
#         frame for frame in stack if filter_keyword in frame.filename
#     ]
    
#     for frame in filtered_stack:
#         print(f'File "{frame.filename}", line {frame.lineno}, in {frame.name}')
#         print(f'  {frame.line}')

def patched_patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
    with self.use_ejected():
        #print_filtered_stack()

        self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=False)
        for k in self.object_patches:
            old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old
       
    self.inject_model()
    return self.model

def patched_load_lora_for_models(model, clip, lora, strength_model, strength_clip):
    patch_keys = list(model.object_patches_backup.keys())
    for k in patch_keys:
        comfy.utils.set_attr(model.model, k, model.object_patches_backup[k])

    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora, key_map)
   
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
    else:
        k1 = ()
        new_clip = None

    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED {}".format(x))

    if patch_keys:
        if hasattr(model.model, "compile_settings"):
            compile_settings = getattr(model.model, "compile_settings")
            for k in patch_keys:
                if "diffusion_model." in k:
                    # Remove the prefix to get the attribute path
                    key = k.replace('diffusion_model.', '')
                    attributes = key.split('.')
                    # Start with the diffusion_model object
                    block = model.get_model_object("diffusion_model")
                    # Navigate through the attributes to get to the block
                    for attr in attributes:
                        if attr.isdigit():
                            block = block[int(attr)]
                        else:
                            block = getattr(block, attr)
                    # Compile the block
                    compiled_block = torch.compile(
                        block,
                        mode=compile_settings["mode"],
                        dynamic=compile_settings["dynamic"],
                        fullgraph=compile_settings["fullgraph"],
                        backend=compile_settings["backend"]
                    )
                    # Add the compiled block back as an object patch
                    model.add_object_patch(k, compiled_block)
    return (new_modelpatcher, new_clip)