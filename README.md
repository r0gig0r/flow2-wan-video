# flow2-wan-video

## Introdution
Work in progress...

## Installation
Search for flow2-wan-video in the Comfyui manager custom node list and click Install.

### Manual installation
1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
2. git clone https://github.com/Flow-two/flow2-wan-video.git
3. Go to `ComfyUI_windows_portable/` folder and Run command to
`python_embeded\python.exe -m pip install -r "ComfyUI\custom_nodes\flow2-wan-video\requirements.txt"`
Done.

### Custom threshold
The **Wan Model Patcher** node now accepts an optional `teacache_rel_l1_thresh` value.
Set this to override the TeaCache relative L1 distance threshold; use `0` for the built-in default of `0.2`.
