from .videohelpersuit.latent_preview import WrappedPreviewer
import threading
import ctypes
import inspect

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

    # thread id of sampling process
    sampling_tid = threading.get_ident()
    timer_lock = threading.Lock()
    watchdog = None

    def _async_raise(tid, exctype):
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def _watchdog_timeout():
        _async_raise(sampling_tid, RuntimeError("Sampling progress appears to be stuck"))

    def _reset_timer(timeout):
        nonlocal watchdog
        with timer_lock:
            if watchdog is not None:
                watchdog.cancel()
            watchdog = threading.Timer(timeout, _watchdog_timeout)
            watchdog.daemon = True
            watchdog.start()

    # allow up to one minute for the first step
    _reset_timer(60)

    def callback(step, x0, x, ts):
        _reset_timer(60 if step == 0 else 30)

        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(x0)
        pbar.update_absolute(step + 1, ts, preview_bytes)

        if getattr(pbar, "total", None) == step + 1:
            with timer_lock:
                if watchdog is not None:
                    watchdog.cancel()

    return callback