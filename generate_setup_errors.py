"""
generate_setup_errors.py
------------------------
Utility module to simulate patient setup errors (interfraction uncertainties).

Uses sub-voxel rigid shifts on 2D density maps via scipy.ndimage.shift.
This allows us to mimic errors in patient positioning relative to the beam isocenter.
"""

import numpy as np
import scipy.ndimage


def apply_rigid_shift(
    density_map: np.ndarray,
    dx_shift_cm: float,
    dy_shift_cm: float,
    voxel_size_cm: float = 0.1
) -> np.ndarray:
    """
    Apply a sub-voxel rigid translation to a 2D map using bilinear interpolation.

    Args:
        density_map:   [ny, nx] 2D array (e.g., density, I-value, Z/A)
        dx_shift_cm:   depth shift (positive = pushes anatomy deeper/rightward)
        dy_shift_cm:   lateral shift (positive = pushes anatomy down/leftward)
        voxel_size_cm: physical size of one voxel (dx = dy)

    Returns:
        shifted_map: [ny, nx] translated 2D array
    """
    # Convert physical shift (cm) to voxel shift
    # scipy.ndimage.shift expects shift vector [shift_y, shift_x] for a 2D array
    shift_voxels = [dy_shift_cm / voxel_size_cm, dx_shift_cm / voxel_size_cm]

    # order=1 means bilinear interpolation.
    # mode='nearest' avoids creating artificial air gaps at the edges by repeating the edge pixel.
    return scipy.ndimage.shift(
        density_map,
        shift_voxels,
        order=1,
        mode='nearest'
    )
