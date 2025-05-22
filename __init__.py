from .custom_nodes import *

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "WanVideoModelPatcher_F2"       : WanVideoModelPatcher_F2,
    "WanVideoModelLoader_F2"        : WanVideoModelLoader_F2,
    "WanI2VModelLoader_F2"        : WanI2VModelLoader_F2,
    "WanVideoConfigure_F2"          : WanVideoConfigure_F2,
    "WanVideoSampler_F2"            : WanVideoSampler_F2,
    "WanVideoEnhancer_F2"           : WanVideoEnhancer_F2,
    "ResizeImage_F2"                : ResizeImage_F2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoModelPatcher_F2"       : "Wan Model Patcher",
    "WanVideoModelLoader_F2"        : "Wan Model Loader",
    "WanI2VModelLoader_F2"        : "Wan I2V Model Loader",
    "WanVideoConfigure_F2"          : "Wan Configure",
    "WanVideoSampler_F2"            : "Wan Sampler",
    "WanVideoEnhancer_F2"           : "Wan Frame Enhancer",
    "ResizeImage_F2"                : "Resize Image",
}