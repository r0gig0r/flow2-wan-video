import torch
import typing
import einops
import gc
import comfy.model_management as mm

def preprocess_frames(frames):
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()

def assert_batch_size(frames, batch_size=2, vfi_name=None):
    subject_verb = "Most VFI models require" if vfi_name is None else f"VFI model {vfi_name} requires"
    assert len(frames) >= batch_size, f"{subject_verb} at least {batch_size} frames to work with, only found {frames.shape[0]}. Please check the frame input using PreviewImage."

def _generic_frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.Union[typing.SupportsInt, typing.List],
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: None,
        use_timestep=True,
        dtype=torch.float16,
    ):

    device = mm.get_torch_device()
    
    #https://github.com/hzwer/Practical-RIFE/blob/main/inference_video.py#L169
    def non_timestep_inference(frame0, frame1, n):        
        middle = return_middle_frame_function(frame0, frame1, None, *return_middle_frame_function_args)
        if n == 1:
            return [middle]
        first_half = non_timestep_inference(frame0, middle, n=n//2)
        second_half = non_timestep_inference(middle, frame1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    output_frames = torch.zeros(multiplier*frames.shape[0], *frames.shape[1:], dtype=dtype, device="cpu")
    out_len = 0

    number_of_frames_processed_since_last_cleared_cuda_cache = 0
    
    for frame_itr in range(len(frames) - 1): # Skip the final frame since there are no frames after it
        frame0 = frames[frame_itr:frame_itr+1]
        output_frames[out_len] = frame0 # Start with first frame
        out_len += 1
        # Ensure that input frames are in fp32 - the same dtype as model
        frame0 = frame0.to(dtype=torch.float32)
        frame1 = frames[frame_itr+1:frame_itr+2].to(dtype=torch.float32)
        
        if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
            continue
    
        # Generate and append a batch of middle frames
        middle_frame_batches = []

        if use_timestep:
            for middle_i in range(1, multiplier):
                timestep = middle_i/multiplier
                
                middle_frame = return_middle_frame_function(
                    frame0.to(device), 
                    frame1.to(device),
                    timestep,
                    *return_middle_frame_function_args
                ).detach().cpu()
                middle_frame_batches.append(middle_frame.to(dtype=dtype))
        else:
            middle_frames = non_timestep_inference(frame0.to(device), frame1.to(device), multiplier - 1)
            middle_frame_batches.extend(torch.cat(middle_frames, dim=0).detach().cpu().to(dtype=dtype))
        
        # Copy middle frames to output
        for middle_frame in middle_frame_batches:
            output_frames[out_len] = middle_frame
            out_len += 1

        number_of_frames_processed_since_last_cleared_cuda_cache += 1
        # Try to avoid a memory overflow by clearing cuda cache regularly
        if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
            mm.soft_empty_cache()
            number_of_frames_processed_since_last_cleared_cuda_cache = 0
        
        gc.collect()

    # Append final frame
    output_frames[out_len] = frames[-1:]
    out_len += 1

    # clear cache for courtesy
    mm.soft_empty_cache()

    return output_frames[:out_len]

def generic_frame_loop(
        model_name,
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.Union[typing.SupportsInt, typing.List],
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: None,
        use_timestep=True,
        dtype=torch.float32):

    assert_batch_size(frames, vfi_name=model_name.replace('_', ' ').replace('VFI', ''))
    if type(multiplier) == int:
        return _generic_frame_loop(
            frames, 
            clear_cache_after_n_frames, 
            multiplier, 
            return_middle_frame_function, 
            *return_middle_frame_function_args, 
            interpolation_states=interpolation_states,
            use_timestep=use_timestep,
            dtype=dtype
        )
    if type(multiplier) == list:
        multipliers = list(map(int, multiplier))
        multipliers += [2] * (len(frames) - len(multipliers) - 1)
        frame_batches = []
        for frame_itr in range(len(frames) - 1):
            multiplier = multipliers[frame_itr]
            if multiplier == 0: continue
            frame_batch = _generic_frame_loop(
                frames[frame_itr:frame_itr+2], 
                clear_cache_after_n_frames, 
                multiplier, 
                return_middle_frame_function, 
                *return_middle_frame_function_args, 
                interpolation_states=interpolation_states,
                use_timestep=use_timestep,
                dtype=dtype,
            )
            if frame_itr != len(frames) - 2: # Not append last frame unless this batch is the last one
                frame_batch = frame_batch[:-1]
            frame_batches.append(frame_batch)
        output_frames = torch.cat(frame_batches)
        return output_frames
    raise NotImplementedError(f"multipiler of {type(multiplier)}")