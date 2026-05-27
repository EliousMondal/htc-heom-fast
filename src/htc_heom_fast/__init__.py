"""Fast matrix-free scaled HEOM tools for symmetric HTC/Tavis-Cummings models.

The package focuses on the first-excitation manifold of an HTC/Tavis-Cummings
system with independent identical site-local Drude-Lorentz baths.
"""

try:
    from .constants import *  # noqa: F401,F403
except Exception:
    pass

__version__ = "0.1.0"
