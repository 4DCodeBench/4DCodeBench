"""Hidden reference-grounded 4DCodeBench scorer."""

import os

# libigl runs single-threaded; the scorer parallelises across frame workers.
os.environ.setdefault("IGL_NUM_THREADS", "1")
