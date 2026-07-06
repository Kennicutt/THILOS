import numpy as np
from numpy.polynomial import polynomial as P
from scipy.interpolate import griddata
import warnings


def directional_polynomial_inpaint(image, bad_mask=None, degree=2, max_radius=7,
                                    directions=None, min_good_pixels=3,
                                    nan_is_bad=True, fill_value=None):
    """
    Parameters
    ----------
    image : ndarray (2D)
        Astronomical image. May contain NaNs in bad pixels (BPM applied).
    bad_mask : ndarray (2D, bool), optional
        Explicit mask of bad pixels. If None, it is derived from NaNs
        (if nan_is_bad=True).
    degree : int
        Degree of the fitting polynomial (1=linear, 2=parabolic).
    max_radius : int
        Maximum search radius for good pixels in each direction.
    directions : list of tuples, optional
        Directions (dy, dx). Default: 8 cardinal+diagonal directions.
    min_good_pixels : int
        Minimum number of good pixels required per direction.
    nan_is_bad : bool
        If True, treats NaNs in `image` as additional bad pixels.
    fill_value : float, optional
        If provided, fills bad pixels with this value instead of
        interpolating. Useful only for flagging purposes.
    
    Returns
    -------
    corrected : ndarray
        Image with bad pixels interpolated. Original NaNs are replaced.
    confidence_map : ndarray
        Confidence map (0-1). NaN where there were no bad pixels.
    bad_mask_combined : ndarray (bool)
        Final combined mask used (NaN + explicit mask).
    """

    image = np.asarray(image, dtype=np.float64)

    # Construir máscara combinada
    if bad_mask is not None:
        bad_mask = np.asarray(bad_mask, dtype=bool)
        combined_mask = bad_mask.copy()
    else:
        combined_mask = np.zeros_like(image, dtype=bool)

    if nan_is_bad:
        nan_mask = np.isnan(image)
        combined_mask = combined_mask | nan_mask

    original_mask = combined_mask.copy()

    if fill_value is not None:
        corrected = image.copy()
        corrected[combined_mask] = fill_value
        confidence_map = np.full_like(image, np.nan)
        confidence_map[combined_mask] = 0.0
        return corrected, confidence_map, combined_mask

    if directions is None:
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

    # Inicializar imagen de trabajo: copiar valores sanos, poner 0 en malos
    corrected = np.where(combined_mask, 0.0, image)
    confidence_map = np.full_like(image, np.nan)
    current_mask = combined_mask.copy()

    iteration = 0
    max_iterations = 30

    while np.any(current_mask) and iteration < max_iterations:
        iteration += 1
        bad_y, bad_x = np.where(current_mask)

        if len(bad_y) == 0:
            break

        source_image = corrected.copy()
        new_corrected = corrected.copy()
        new_mask = current_mask.copy()
        corrected_any = False

        for idx in range(len(bad_y)):
            y, x = bad_y[idx], bad_x[idx]

            estimates = []
            weights = []

            for dy, dx in directions:
                dists = []
                vals = []

                for r in range(1, max_radius + 1):
                    ny = y + dy * r
                    nx = x + dx * r

                    if ny < 0 or ny >= image.shape[0] or nx < 0 or nx >= image.shape[1]:
                        break

                    # Saltar si el vecino aún no ha sido corregido
                    if current_mask[ny, nx]:
                        continue

                    dists.append(float(r))
                    vals.append(source_image[ny, nx])

                    if len(dists) >= degree + 4:
                        break

                if len(dists) < min_good_pixels:
                    continue

                try:
                    dists_arr = np.array(dists, dtype=np.float64)
                    vals_arr = np.array(vals, dtype=np.float64)

                    # Pesos decrecientes con distancia
                    w = np.exp(-dists_arr / (max_radius / 2.0))
                    w = w / np.sum(w)

                    # Ajuste polinomial ponderado
                    result = P.polyfit(dists_arr, vals_arr, degree, w=w, full=True)
                    coeffs = result[0]
                    details = result[1]
                    residuals = details[0] if len(details) > 0 else np.array([0])

                    pred = P.polyval(0, coeffs)

                    n_points = len(dists_arr)
                    max_dist = np.max(dists_arr)

                    conf_n = min(n_points / (degree + 3), 1.0)

                    if residuals.size > 0 and np.sum(residuals) > 1e-12:
                        data_var = np.var(vals_arr)
                        if data_var > 1e-12:
                            conf_fit = np.exp(-np.sum(residuals) / (data_var * n_points * 0.5))
                        else:
                            conf_fit = 0.5
                    else:
                        conf_fit = 1.0
                    conf_fit = np.clip(conf_fit, 0.05, 1.0)

                    conf_dist = np.exp(-max_dist / max_radius)
                    confidence = conf_n * conf_fit * conf_dist

                    estimates.append(pred)
                    weights.append(confidence)

                except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                    continue

            if len(estimates) == 0:
                continue

            weights = np.array(weights)
            estimates = np.array(estimates)
            w_norm = weights / np.sum(weights)
            final_value = np.sum(estimates * w_norm)
            total_confidence = np.sum(weights * w_norm)
            total_confidence = np.clip(total_confidence, 0, 1)

            new_corrected[y, x] = final_value
            confidence_map[y, x] = total_confidence
            new_mask[y, x] = False
            corrected_any = True

        corrected = new_corrected
        current_mask = new_mask

        if not corrected_any:
            break

    # Fallback para píxeles no corregidos
    if np.any(current_mask):
        y_good, x_good = np.where(~original_mask)
        y_bad, x_bad = np.where(current_mask)
        if len(y_good) > 0 and len(y_bad) > 0:
            values_good = corrected[y_good, x_good]
            values_interp = griddata(
                (y_good, x_good), values_good,
                (y_bad, x_bad), method='linear'
            )
            nan_interp = np.isnan(values_interp)
            if np.any(nan_interp):
                values_interp[nan_interp] = griddata(
                    (y_good, x_good), values_good,
                    (y_bad[nan_interp], x_bad[nan_interp]), method='nearest'
                )
            corrected[y_bad, x_bad] = values_interp
            confidence_map[y_bad, x_bad] = 0.2

    return corrected, confidence_map, combined_mask