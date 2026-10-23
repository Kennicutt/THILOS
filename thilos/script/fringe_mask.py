import numpy as np
import sep
from scipy import ndimage
from scipy.optimize import minimize_scalar
from scipy.interpolate import griddata
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog 
from photutils.background import Background2D, MedianBackground

from loguru import logger
import warnings
import logging, inspect

############################################################################################
# Add the logger
############################################################################################

class InterceptHandler(logging.Handler):
    """Custom logging handler that redirects log records from the standard logging module 
    to the Loguru logger.

    This handler allows the use of Loguru alongside the standard logging module, ensuring 
    that logs captured by the standard logging system are properly forwarded to the Loguru logger.

    Args:
        logging (module): The logging module that provides the log records to be captured 
                           and forwarded to Loguru. Typically, this would be the standard 
                           `logging` module.
    """
    
    def emit(self, record: logging.LogRecord):
        """
        Process a log record and forward it to the Loguru logger, mapping the log level and 
        determining the caller's origin for the log message.

        This method retrieves the log level, the caller's information, and the log message 
        before passing it to Loguru for proper logging.

        Args:
            record (logging.LogRecord): The log record to be processed, which contains 
                                        information about the log level, message, and other 
                                        relevant details for the log entry.

        This method will convert the logging level to a corresponding Loguru level and 
        propagate the message along with the exception information (if any) to the Loguru logger.
        """
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        if level != "DEBUG":
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=False)




def log_and_raise(exc: Exception, *, level: str = "ERROR"):
    """Logs an exception and raises it.

    Args:
        exc (Exception): The exception to be logged and raised.
        level (str, optional): The logging level to use. Defaults to "ERROR".
    """
    logger.bind().log(level, f"Caught exception: {exc}", exception=exc)
    raise

#########################################################################################
# Start the code to defringe
#########################################################################################

def create_fringe_mask(images, threshold_sigma=2.0, npixels=5, 
                       deblend=True, dilation_radius=5,
                       min_area=10, edge_mask_width=50,
                       method='interpolation', verbose=True):
    """
    Crea máscaras de objetos y genera un fringe map limpio a partir de un stack.
    
    Parameters
    ----------
    images : list o ndarray
        Lista de arrays 2D o ndarray de shape (N, height, width)
    threshold_sigma : float
        Umbral de detección en sigma sobre el fondo
    npixels : int
        Mínimo de píxeles conectados para considerar una fuente
    deblend : bool
        Si True, separa fuentes superpuestas
    dilation_radius : int
        Radio para dilatar la máscara (capturar halos y PSF)
    min_area : int
        Área mínima de fuente para ser enmascarada
    edge_mask_width : int
        Píxeles a enmascarar en los bordes (vignetting, artefactos)
    method : str
        'interpolation' o 'median' para rellenar zonas enmascaradas
    verbose : bool
    
    Returns
    -------
    fringe_map : ndarray
        Fringe map limpio listo para restar
    master_mask : ndarray
        Máscara combinada de todas las fuentes detectadas
    individual_masks : list
        Máscaras de cada imagen individual
    """
    
    # Convertir a array si es lista
    if isinstance(images, list):
        images = np.array(images, dtype=float)
    
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    
    n_images, ny, nx = images.shape
    individual_masks = []
    
    # ============================================
    # PASO 1: Detección de fuentes en cada imagen
    # ============================================
    
    for i, img in enumerate(images):
        if verbose:
            print(f"Procesando imagen {i+1}/{n_images}...")
        
        # Estadísticas del fondo con sigma clipping
        mean, median, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
        
        # Background 2D para manejar gradientes grandes
        try:
            sigma_clip_bg = SigmaClip(sigma=3.0, maxiters=3)
            bkg_estimator = MedianBackground()
            bkg = Background2D(img, (64, 64), filter_size=(3, 3),
                              sigma_clip=sigma_clip_bg, bkg_estimator=bkg_estimator)
            background = bkg.background
            rms = bkg.background_rms
        except Exception:
            # Fallback: fondo plano
            background = median
            rms = std
        
        # Imagen sin fondo para detección
        data_sub = img - background
        
        # Umbral de detección
        if np.isscalar(rms):
            threshold = threshold_sigma * rms
        else:
            threshold = threshold_sigma * rms
        
        # Detección de fuentes
        segm = detect_sources(data_sub, threshold, npixels=npixels)
        
        if segm is None:
            mask = np.zeros_like(img, dtype=bool)
        else:
            # Deblending de fuentes superpuestas
            if deblend and segm.nlabels > 0:
                segm_deblend = deblend_sources(data_sub, segm, npixels=npixels,
                                               nlevels=32, contrast=0.001)
            else:
                segm_deblend = segm
            
            # ============================================
            # CORRECCIÓN: API actual de photutils
            # ============================================
            cat = SourceCatalog(data_sub, segm_deblend)
            
            # Obtener áreas de forma compatible con la API actual
            # cat.area puede ser un array, lista, o atributo especial
            try:
                # Intentar como array directo
                areas = np.array(cat.area)
            except (TypeError, AttributeError):
                try:
                    # Intentar como columna de tabla
                    areas = cat.to_table()['area'].data
                except Exception:
                    # Último recurso: calcular manualmente
                    areas = np.array([np.sum(segm_deblend.data == label) 
                                      for label in segm_deblend.labels])
            
            # Crear máscara de fuentes grandes
            labels_to_mask = []
            for j, label in enumerate(segm_deblend.labels):
                area = areas[j] if j < len(areas) else 0
                if area >= min_area:
                    labels_to_mask.append(label)
            
            mask = np.isin(segm_deblend.data, labels_to_mask)
            
            # Dilatar máscara para capturar halos y PSF
            if dilation_radius > 0:
                mask = ndimage.binary_dilation(mask, 
                                               iterations=dilation_radius)
        
        # Enmascarar bordes
        mask[:edge_mask_width, :] = False
        mask[-edge_mask_width:, :] = False
        mask[:, :edge_mask_width] = False
        mask[:, -edge_mask_width:] = False
        
        # Enmascarar NaNs e Inf
        mask |= ~np.isfinite(img)
        
        individual_masks.append(mask)
    
    # ============================================
    # PASO 2: Combinar máscaras
    # ============================================
    
    # Máscara maestra: un píxel enmascarado en CUALQUIER imagen queda enmascarado
    master_mask = np.any(individual_masks, axis=0)
    
    if verbose:
        print(f"Porcentaje enmascarado: {100*np.sum(master_mask)/master_mask.size:.1f}%")
    
    # ============================================
    # PASO 3: Crear fringe map
    # ============================================
    
    # Aplicar máscara maestra a todas las imágenes
    masked_stack = np.where(master_mask, np.nan, images)
    
    # Fringe map: mediana del stack en píxeles no enmascarados
    fringe_map_raw = np.nanmedian(masked_stack, axis=0)
    
    # ============================================
    # PASO 4: Rellenar zonas enmascaradas
    # ============================================
    
    if method == 'interpolation':
        fringe_map = _interpolate_masked(fringe_map_raw, master_mask)
    elif method == 'median':
        fringe_map = _median_fill(fringe_map_raw, master_mask, size=51)
    else:
        fringe_map = fringe_map_raw.copy()
        fringe_map[master_mask] = np.nanmedian(fringe_map_raw[~master_mask])
    
    # Normalizar: fringe map debe tener media ~ 0 (es una corrección)
    fringe_map = fringe_map - np.nanmedian(fringe_map)
    
    if verbose:
        print("Fringe map creado exitosamente.")
    
    return fringe_map, master_mask, individual_masks


def _interpolate_masked(data, mask, method='linear'):
    """Interpola valores en zonas enmascaradas."""
    ny, nx = data.shape
    y, x = np.mgrid[0:ny, 0:nx]
    
    # Píxeles válidos
    valid = ~mask
    if np.sum(valid) < 10:
        warnings.warn("Muy pocos píxeles válidos para interpolación")
        return data
    
    # Interpolación
    points = np.column_stack((y[valid], x[valid]))
    values = data[valid]
    grid = (y, x)
    
    filled = griddata(points, values, grid, method=method, fill_value=np.nanmedian(data))
    
    # Rellenar NaNs residuales con mediana
    filled[np.isnan(filled)] = np.nanmedian(data)
    
    return filled


def _median_fill(data, mask, size=51):
    """Rellena con filtro de mediana iterativo."""
    filled = data.copy()
    filled[mask] = np.nan
    
    # Kernel grande para propagar valores
    kernel = np.ones((size, size))
    
    for _ in range(5):
        # Mediana local ignorando NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_filtered = ndimage.generic_filter(filled, np.nanmedian, 
                                                     footprint=kernel,
                                                     mode='reflect')
        filled[mask] = median_filtered[mask]
        
        # Verificar convergencia
        if np.all(np.isfinite(filled[mask])):
            break
    
    return filled

#=========================================================================================
# Function for defringing correction in the case of NO DITHERING images
#=========================================================================================
"""
defringe_z.py - Correccion de fringing para imagenes con filtro Sloan z.

Modelo:  I(x,y) = k * F(x,y) + a0 + a1*xn + a2*yn  (+ terminos de mayor orden opcionales)

donde F es el masterfringe (que incluye su propio cielo), k es el factor de escala
del patron de fringe, y (a0, a1, a2) absorben la DIFERENCIA de cielo de fondo entre
la imagen cientifica y el masterfringe (offset + gradiente 2D).
"""
import numpy as np
from scipy.optimize import least_squares


def _sigclip_mask(r, sigma=3.0, iters=3):
    m = np.ones_like(r, dtype=bool)
    for _ in range(iters):
        med = np.median(r[m])
        mad = np.median(np.abs(r[m] - med)) * 1.4826
        m2 = np.abs(r - med) < sigma * mad
        if m2.sum() == m.sum():
            break
        m = m2
    return m


def fit_defringe(science, master_fringe, sky_deg=0, niter=5, gain=1.0,
                 source_mask=None, verbose=True):
    """
    Ajusta el masterfringe a la imagen cientifica.

    Parameters
    ----------
    science, master_fringe : array 2D (mismas dimensiones)
    sky_deg : int (0, 1 o 2) - grado del polinomio bivariado que modela la
              diferencia de cielo. 0 = offset constante, 1 = plano, 2 = cuadratico.
    gain : electrons/ADU para pesos poissonianos (si no se conoce, usar 1).
    source_mask : array booleana 2D opcional; True = pixel contaminado por
                  fuentes (se excluye del ajuste). Si es None, se detecta por
                  sigma-clipping iterado.

    Returns
    -------
    dict con: corrected (imagen corregida), k, k_err, sky_params, sky_err,
               chi2_red, r2, rms, n_pix_used, fit_mask
    """
    ny, nx = science.shape
    y, x = np.mgrid[0:ny, 0:nx]
    xn = (x.ravel() - nx / 2) / nx   # coordenadas normalizadas
    yn = (y.ravel() - ny / 2) / ny
    F = master_fringe.ravel().astype(float)
    X = science.ravel().astype(float)

    # Base polinomica bivariada para el cielo
    terms = [np.ones_like(xn)]
    if sky_deg >= 1:
        terms += [xn, yn, xn * yn]
    if sky_deg >= 2:
        terms += [xn**2, yn**2, xn**2 * yn, xn * yn**2, xn**2 * yn**2]
    A = np.column_stack([F] + terms)
    npar = A.shape[1]

    def resid(p, m):
        model = A[m] @ p
        return (X[m] - model) / np.sqrt(np.abs(model) / gain + 1.0)

    mask = np.ones_like(X, dtype=bool)
    if source_mask is not None:
        mask = ~source_mask.ravel().astype(bool)

    for _ in range(niter):
        sol = least_squares(resid, [1.0] + [0.0] * (npar - 1), args=(mask,), method='lm')
        r = X[mask] - A[mask] @ sol.x
        keep = _sigclip_mask(r, sigma=3.0)
        if keep.sum() == mask.sum():
            break
        mask[mask] = keep

    p = sol.x
    m = mask
    dof = m.sum() - npar
    residuos = X[m] - A[m] @ p

    # Errores de parametros: covarianza a partir del jacobiano, escalada por chi2 reducido
    _, s, VT = np.linalg.svd(sol.jac, full_matrices=False)
    cov = VT.T @ np.diag(1.0 / s**2) @ VT
    perr = np.sqrt(np.diag(cov)) * np.sqrt(np.sum(residuos**2) / dof)

    # Bondad del ajuste
    chi2 = np.sum(residuos**2 / (np.abs(A[m] @ p) / gain + 1.0))
    chi2_red = chi2 / dof
    r2 = 1 - np.sum(residuos**2) / np.sum((X[m] - X[m].mean())**2)
    rms = np.sqrt(np.mean(residuos**2))

    # Imagen corregida: restamos k*F + superficie de cielo ajustada
    model_full = A @ p
    corrected = (X - model_full).reshape(ny, nx)

    if verbose:
        names = ['k (escala fringe)'] + ['cielo c%d' % i for i in range(len(terms))]
        print('=== Ajuste de defringing ===')
        for n, v, e in zip(names, p, perr):
            print(f'  {n:20s} = {v:10.4f} +/- {e:.4f}')
        print(f'  chi2 reducido = {chi2_red:.3f}   (ideal ~1; si es <<1 los pesos son pesimistas)')
        print(f'  R2            = {r2:.4f}    (fraccion de varianza explicada)')
        print(f'  RMS residuos  = {rms:.2f} ADU   (debe ser ~ ruido de lectura+cielo)')
        print(f'  pixeles usados= {m.sum()}/{X.size} ({100*m.sum()/X.size:.1f}%)')

    return dict(corrected=corrected, k=p[0], k_err=perr[0],
                sky_params=p[1:], sky_err=perr[1:],
                chi2_red=chi2_red, r2=r2, rms=rms,
                n_pix_used=int(m.sum()), fit_mask=m.reshape(ny, nx))


############################## SOLUTION WITH DITHERING IMAGES ##############################

def get_mask(mask):
    matrix = np.ones(mask.shape)
    matrix[mask == True] = np.nan
    return matrix

def defringe_with_dithering(lst_frame):

    __, __, individual_masks = create_fringe_mask(lst_frame, edge_mask_width=1)

    lst_out_illumination = []
    for i, frame in enumerate(lst_frame):
        frame = frame.astype(frame.dtype.newbyteorder('='))
        bkg = sep.Background(frame, mask=individual_masks[i], bw=64, bh=64, fw=3, fh=3)
        lst_out_illumination.append(frame/(bkg.back()/np.nanmedian(bkg.back())))


    lst_bkg_illu_ok = []
    for i, frame in enumerate(lst_out_illumination):
        n_bkg = sep.Background(frame, mask=individual_masks[i], bw=64, bh=64, fw=3, fh=3)
        lst_bkg_illu_ok.append(n_bkg.back())

    fringe_maps = []
    for i, frame in enumerate(lst_frame):
        MASK = get_mask(individual_masks[i])
        fringe_maps.append((frame - lst_bkg_illu_ok[i]) * MASK)

    return np.nanmedian(fringe_maps, axis=0)

################################### SOLUTION WITHOUT DITHERING IMAGES ####################################

"""
defringe_z.py - Correccion de fringing (Sloan z) SIN sustraer el cielo de fondo.

Problema: el masterfringe incluye su propio cielo. Si se resta escalado,
se elimina tambien el cielo de la imagen cientifica.

Solucion:
  1. Se estima el nivel de cielo del masterfringe (mediana sigma-clipped)
     y se construye el patron puro:  P = master - <master>   (media ~0)
  2. Se ajusta  I = k*P + a0 + a1*xn + a2*yn (+ terminos de mayor orden)
     donde los terminos polinomicos son MODESTOS: solo estabilizan la
     estimacion de k ante gradientes de cielo, NO se restan de la imagen.
  3. Correccion:  I_corr = I - k*P    -> el cielo queda intacto.

Ajuste: minimos cuadrados ponderados (pesos de los datos, 1/sqrt(|I|)),
lineales y de solucion cerrada, con sigma-clipping iterado (3-sigma, MAD)
para excluir fuentes. Determinista: sin minimos locales.
"""
import numpy as np


def _sigclip_median(d, sigma=3.0, iters=5):
    m = np.ones_like(d, dtype=bool)
    for _ in range(iters):
        med = np.median(d[m])
        mad = np.median(np.abs(d[m] - med)) * 1.4826
        m2 = np.abs(d - med) < sigma * mad
        if m2.sum() == m.sum():
            break
        m = m2
    return np.median(d[m]), m


def fit_defringe_no_dithering(science, master_fringe, sky_deg=1, niter=8, gain=1.0,
                 source_mask=None, verbose=True):
    """
    Ajusta y corrige el fringing sin tocar el cielo de fondo.

    Parameters
    ----------
    science, master_fringe : array 2D (mismas dimensiones)
    sky_deg : 0, 1 o 2. Grado del polinomio bivariado *molesto* que modela
              la diferencia de cielos durante el ajuste. NUNCA se resta.
    gain : e-/ADU para los pesos poissonianos (1.0 si se desconoce).
    source_mask : bool 2D opcional (True = pixel con fuente, excluir).

    Returns
    -------
    dict: corrected, k, k_err, chi2_red, r2, rms, sky_shift, fit_mask,
          fringe_pattern (P), sky_master
    """
    ny, nx = science.shape
    y, x = np.mgrid[0:ny, 0:nx]
    xn = (x.ravel() - nx / 2) / nx
    yn = (y.ravel() - ny / 2) / ny

    # 1) Patron de fringe con media cero
    sky_master, _ = _sigclip_median(master_fringe.ravel())
    P2d = master_fringe - sky_master
    P2d -= P2d.mean()   # media exactamente 0: la correccion no desplaza el cielo global
    F = P2d.ravel().astype(float)
    X = science.ravel().astype(float)

    # 2) Disenho: [k | polinomio de cielo (modesto)]
    terms = [np.ones_like(xn)]
    if sky_deg >= 1:
        terms += [xn, yn, xn * yn]
    if sky_deg >= 2:
        terms += [xn**2, yn**2, xn**2 * yn, xn * yn**2, xn**2 * yn**2]
    A = np.column_stack([F] + terms)
    npar = A.shape[1]

    # 3) Ajuste ponderado lineal + sigma-clipping iterado
    w = 1.0 / np.sqrt(np.abs(X) / gain + 1.0)
    m = np.ones_like(X, dtype=bool)
    if source_mask is not None:
        m = ~source_mask.ravel().astype(bool)
    for _ in range(niter):
        p, _, _, _ = np.linalg.lstsq(A[m] * w[m, None], X[m] * w[m], rcond=None)
        r = X[m] - A[m] @ p
        keep = np.abs(r - np.median(r)) < 3.0 * np.median(np.abs(r - np.median(r))) * 1.4826
        if keep.sum() == m.sum():
            break
        m[m] = keep

    dof = m.sum() - npar
    resid = X[m] - A[m] @ p
    chi2_red = np.sum((resid * w[m])**2) / dof
    r2 = 1 - np.sum(resid**2) / np.sum((X[m] - X[m].mean())**2)

    # Errores: cov = s^2 (A' W A)^-1
    Aw = A * w[:, None]
    cov = np.linalg.inv(Aw[m].T @ Aw[m]) * np.sum(resid**2) / dof
    perr = np.sqrt(np.diag(cov))

    # 4) Correccion: SOLO el patron escalado. El cielo NO se toca.
    corrected = science - p[0] * P2d

    # Verificacion de preservacion del cielo
    sky_antes, _ = _sigclip_median(X[m])
    sky_desp, _ = _sigclip_median(corrected.ravel()[m])

    if verbose:
        print('=== Defringing (cielo preservado) ===')
        print(f'  k (escala fringe) = {p[0]:.4f} +/- {perr[0]:.4f}')
        print(f'  cielo master      = {sky_master:.2f} ADU')
        print(f'  chi2 reducido     = {chi2_red:.3f}   R2 = {r2:.4f}')
        print(f'  RMS residuos      = {np.sqrt(np.mean(resid**2)):.2f} ADU')
        print(f'  delta cielo global= {sky_desp - sky_antes:+.3f} ADU (debe ser ~0)')
        print(f'  pixeles usados    = {m.sum()}/{X.size} ({100*m.sum()/X.size:.1f}%)')

    return dict(corrected=corrected, k=p[0], k_err=perr[0],
                chi2_red=chi2_red, r2=r2, rms=float(np.sqrt(np.mean(resid**2))),
                sky_shift=float(sky_desp - sky_antes), fit_mask=m.reshape(ny, nx),
                fringe_pattern=P2d, sky_master=float(sky_master))

################################################ CREATE MASK FROM SCIENCE FRAME ####################################################

def create_mask_from_science(science, threshold_sigma=2.0, npixels=5, 
                       deblend=True, dilation_radius=5,
                       min_area=10, edge_mask_width=0,
                       verbose=True):
    """
    Crea una máscara de objetos a partir de una imagen científica.

    Parameters
    ----------
    science : ndarray
        Imagen científica 2D
    threshold_sigma : float
        Umbral de detección en sigma sobre el fondo
    npixels : int
        Mínimo de píxeles conectados para considerar una fuente
    deblend : bool
        Si True, separa fuentes superpuestas
    dilation_radius : int
        Radio para dilatar la máscara (capturar halos y PSF)
    min_area : int
        Área mínima de fuente para ser enmascarada
    edge_mask_width : int
        Píxeles a enmascarar en los bordes (vignetting, artefactos)
    verbose : bool
    
    Returns
    -------
    mask : ndarray
        Máscara booleana de la imagen científica (True = pixel contaminado)
    """
    
    # Estadísticas del fondo con sigma clipping
    mean, median, std = sigma_clipped_stats(science, sigma=3.0, maxiters=5)
    
    # Background 2D para manejar gradientes grandes
    try:
        sigma_clip_bg = SigmaClip(sigma=3.0, maxiters=3)
        bkg_estimator = MedianBackground()
        bkg = Background2D(science, (64, 64), filter_size=(3, 3),
                          sigma_clip=sigma_clip_bg, bkg_estimator=bkg_estimator)
        background = bkg.background
        rms = bkg.background_rms
    except Exception:
        # Fallback: fondo plano
        background = median
        rms = std
    
    # Imagen sin fondo para detección
    data_sub = science - background
    
    # Umbral de detección
    if np.isscalar(rms):
        threshold = threshold_sigma * rms
    else:
        threshold = threshold_sigma * rms
    
    # Detección de fuentes
    segm = detect_sources(data_sub, threshold, npixels=npixels)

    if segm is None:
        mask = np.zeros_like(science, dtype=bool)
    else:
        # Deblending de fuentes superpuestas
        if deblend and segm.nlabels > 0:
            segm_deblend = deblend_sources(data_sub, segm, npixels=npixels,
                                           nlevels=32, contrast=0.001)
        else:
            segm_deblend = segm
        
        # ============================================
        # CORRECCIÓN: API actual de photutils
        # ============================================
        cat = SourceCatalog(data_sub, segm_deblend)
        
        # Obtener áreas de forma compatible con la API actual
        try:
            areas = np.array(cat.area)
        except (TypeError, AttributeError):
            try:
                areas = cat.to_table()['area'].data
            except Exception:
                areas = np.array([np.sum(segm_deblend.data == label) 
                                  for label in segm_deblend.labels])
        
        # Crear máscara de fuentes grandes
        labels_to_mask = []
        for j, label in enumerate(segm_deblend.labels):
            area = areas[j] if j < len(areas) else 0
            if area >= min_area:
                labels_to_mask.append(label)
        
        mask = np.isin(segm_deblend.data, labels_to_mask)
        
        # Dilatar máscara para capturar halos y PSF
        if dilation_radius > 0:
            mask = ndimage.binary_dilation(mask, 
                                           iterations=dilation_radius)

    # Enmascarar bordes
    if edge_mask_width > 0:
        mask = mask | np.pad(mask, pad_width=edge_mask_width, mode='constant', constant_values=False)

    # Enmascarar NaNs e Inf
    mask |= ~np.isfinite(science)

    if verbose:
        print(f"Porcentaje enmascarado: {100*np.sum(mask)/mask.size:.1f}%")

    return mask

########################################### Illumination correction #########################################################

def illu_corr(frame):
    mask = create_mask_from_science(frame)

    bkg = sep.Background(frame, mask=mask,
                         bw=64, bh=64, fw=3, fh=3)

    return frame/(bkg.back()/np.nanmedian(bkg.back()))



#########################################################################################
# Defringe No Dithering
#########################################################################################

def make_source_mask(image, nsigma=3.0, iters=3):
    """
    Máscara de píxeles libres de fuentes: itera mediana + corte sigma.
    True = fondo puro (útil para el ajuste).
    """
    mask = np.ones_like(image, dtype=bool)
    for _ in range(iters):
        med = np.median(image[mask])
        mad = np.median(np.abs(image[mask] - med))
        mask = np.abs(image - med) < nsigma * 1.4826 * mad
    return mask

def norm(frame):
    return (frame - np.nanmin(frame))/(np.nanmax(frame) - np.nanmin(frame))

def defringe_no_dithering(frame, masterfringe):

    #Create the mask to remove sources
    MASK = make_source_mask(frame)

    #Substract background to science frame
    frame = frame.astype(frame.dtype.newbyteorder('='))
    bkg = sep.Background(frame, bw=64, bh=64, fw=3, fh=3)
    frame_no_sky = frame.copy() - bkg.back()

    MSCI = frame_no_sky.copy()
    MFRI = masterfringe.copy()

    MSCI[~MASK] = np.nan
    MFRI[~MASK] = np.nan

    MSCI = norm(MSCI)
    MFRI = norm(MFRI)

    #Estimate scale factor
    k = MSCI / MFRI

    #Estimate global scale factor
    k_global = np.nanmedian(k)
    logger.info(f"k global value: {k_global}")

    k_global_corr = k_global / (10 ** np.ceil(np.log10(k_global)))
    logger.info(f"k global corrected value: {k_global_corr}")

    #Compute residuals
    residuals = MSCI - (k_global * MFRI)

    #Compute Chi-squared
    chi2_min = np.nansum(residuals**2 / (k_global * MFRI))
    logger.info(fr"$\CHI²$: {chi2_min}")

    #Estimate ddof
    ddof = len(MSCI[MASK]) - 1 # 1 parameter used

    #Compute Chi2-reduced
    chi2_reduced = chi2_min / ddof
    logger.info(fr"$\CHI²$ reduced: {chi2_reduced}")

    #Compute RMS
    rms = np.nanstd(residuals)
    logger.info(f"Residual rms: {rms} adus")

    #Compute median
    med = np.nanmedian(residuals)
    logger.info(f"Residual median: {med} adus")

    #Apply k global corrected to science
    corrected_frame = frame - (k_global_corr * masterfringe)

    return {"k_global": k_global, "k_global_corr": k_global_corr, "Chi-2": chi2_min,
    "Chi-2_Red": chi2_reduced, "RMS": rms, "Median": med, "result": corrected_frame}
