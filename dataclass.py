from dataclasses import dataclass

@dataclass
class Config:
    positive: str
    negative: str
    width: int
    height: int
    frames: float
    guidance_scale: float
    guidance_percent: float
    flow_shift: float
    sampling_steps: int
    enhance_strength: float
    cfg_zero_steps: int
    skip_layer: str
    skip_start_percent: float
    skip_end_percent: float
    extend_video_count: int
