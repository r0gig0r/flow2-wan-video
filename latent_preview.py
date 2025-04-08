from .videohelpersuit.latent_preview import WrappedPreviewer

class LatentPreviewer:
    pass

class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

def get_previewer(taesd, device, resolution):
    previewer = TAESDPreviewerImpl(taesd.to(device))
    previewer = WrappedPreviewer(previewer, rate=16, resolution=resolution)
    return previewer

def prepare_callback(previewer, pbar, x0_output_dict=None):
    def callback(step, x0, x, ts):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(x0)
        pbar.update_absolute(step + 1, ts, preview_bytes)
    return callback