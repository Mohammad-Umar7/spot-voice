"""spot-voice: hands-free, voice-commanded operation of a Boston Dynamics Spot."""

import os

__version__ = "1.0.0"

# ----------------------------------------------------------------------
# OpenMP duplicate-runtime guard.
#
# faster-whisper (via ctranslate2) ships its own OpenMP runtime, and so do
# several of the numeric wheels this project already depends on. When two get
# loaded into one process, Intel's runtime aborts the program outright:
#
#     OMP: Error #15: Initializing libiomp5md.dll, but found
#     libiomp5md.dll already initialized.
#
# That kills the voice loop on the first transcription. Setting this before any
# of those libraries load lets the duplicate through.
#
# Intel calls the flag unsupported, and the risk it names is threading
# misbehaviour under heavy parallel numeric work. Pinning the thread count to 1
# sidesteps that: the two workloads here -- a small Whisper model and YOLOv8n --
# are short and run on their own threads anyway, so there is nothing to gain
# from intra-op parallelism and one less way for it to go wrong.
#
# Both are set with setdefault, so anything already in the environment wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__all__ = ["__version__"]
