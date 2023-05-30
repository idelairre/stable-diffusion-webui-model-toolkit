from threading import Thread

from modules import shared
from scripts.autoprune import autoprune
from scripts.constants import AUTOPRUNE_PATH


if hasattr(shared.opts, "model_toolkit_autoprune") and shared.opts.model_toolkit_autoprune:
    autoprune_thread = Thread(target=autoprune, args=[AUTOPRUNE_PATH])
    autoprune_thread.start()
    