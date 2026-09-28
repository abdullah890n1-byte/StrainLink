"""StrainLink: geotechnical DIC with sensor fusion and evidence-based correlation."""

import os as _os

# StrainLink parallelises its own work across CPU cores. If the linear-algebra
# library (Apple Accelerate / OpenBLAS / OpenMP) also starts threads inside
# each of those workers, the two thread pools fight: in testing the DIC froze
# for up to 16 minutes at random photos. One library thread per worker avoids
# it. These must be set before numpy is first imported.
for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "0.1.0"
