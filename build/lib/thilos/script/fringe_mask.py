import numpy as np
from scipy import ndimage
from scipy.optimize import minimize_scalar
from scipy.interpolate import griddata
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from photutils.background import Background2D, MedianBackground
import warnings

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
        mask[:edge_mask_width, :] = True
        mask[-edge_mask_width:, :] = True
        mask[:, :edge_mask_width] = True
        mask[:, -edge_mask_width:] = True
        
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