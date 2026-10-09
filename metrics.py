"""Classical P.910 SI/TI on original 8-bit Y code values (not 2023 HDR)."""
import numpy as np
from scipy.ndimage import sobel


def spatial_information(y):
    y = np.asarray(y, dtype=np.float64)
    gx = sobel(y, axis=1)[1:-1, 1:-1]
    gy = sobel(y, axis=0)[1:-1, 1:-1]
    return float(np.std(np.hypot(gx, gy), ddof=0))


def temporal_information(current, previous):
    return float(np.std(np.asarray(current, dtype=np.float64) -
                        np.asarray(previous, dtype=np.float64), ddof=0))


def psnr_from_mse(mse):
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255.0 ** 2 / mse))


def pq_luma(y, color_range='limited'):
    """P.910 (2023) SDR pipeline: limited 8 bit, BT.1886, 300/.01 nits, PQ.

    BT.1886 uses its full black-offset formula, not the simplified display
    model (linear rescaling of x**2.4) used by siti-tools' default pipeline.
    Values outside [16,235] are clipped for the complexity computation only.
    """
    if color_range not in ['limited', 'full']:
        raise ValueError(color_range)
    y = np.asarray(y, dtype=np.float64)
    value = np.clip((y - 16) / 219 if color_range=='limited' else y/255, 0, 1)
    white, black, gamma = 300.0, .01, 2.4
    luminance = ((white ** (1/gamma) - black ** (1/gamma)) * value + black ** (1/gamma)) ** gamma
    power = (luminance / 10000) ** (2610 / 16384)
    return 255 * ((3424/4096 + (2413/128) * power) / (1 + (2392/128) * power)) ** (2523/32)
