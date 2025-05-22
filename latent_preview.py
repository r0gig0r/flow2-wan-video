from .videohelpersuit.latent_preview import WrappedPreviewer
import signal

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
    """Return sampling callback with timeout detection."""

    def _timeout_handler(signum, frame):
        raise RuntimeError("Sampling progress appears to be stuck")

    # allow up to one minute for the first step
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(60)

    def callback(step, x0, x, ts):
        # reset timer: first step can take longer
        signal.alarm(60 if step == 0 else 30)

        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(x0)
        pbar.update_absolute(step + 1, ts, preview_bytes)

        # cancel timer when finished
        if getattr(pbar, "total", None) == step + 1:
            signal.alarm(0)

    return callback
