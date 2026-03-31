"""
================================================================================
ALGORITMO DE ALINEAMIENTO DE IMÁGENES ASTRONÓMICAS
Basado en identificación de centroides estelares
================================================================================

Características principales:
- Detección automática de estrellas mediante centroide de masa ponderado
- Funciona con 1 estrella (traslación simple) o múltiples estrellas (transformación rígida/afín)
- Apilado con múltiples métodos (media, mediana, sigma clipping)
- Sub-pixel accuracy en el alineamiento
"""

import os, glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import CCDData
from astropy import units as u
from skimage import transform as tf
import sep
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform, AffineTransform
from itertools import combinations

from skimage.transform import warp
from skimage import io

from scipy.spatial.distance import cdist

from pathlib import Path
import numpy as np
import cv2, os
from scipy import ndimage
from scipy.optimize import minimize
import ccdproc as ccdp
from astropy.table import Table
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from loguru import logger
import warnings

from THILOS.Color_Codes import bcolors as bcl

import logging, inspect

#Define constants
#Using SEP to detect sources in the reference image
kernel = np.array([[1., 2., 3., 2., 1.],
                   [2., 3., 5., 3., 2.],
                   [3., 5., 8., 5., 3.],
                   [2., 3., 5., 3., 2.],
                   [1., 2., 3., 2., 1.]])
#?kernel = np.ascontiguousarray(kernel) #!
visualize = False
#Define functions

#Create function to visualize the images using ZScale contrast
def visualize_image(data, title, wcs=None, block=False, time=3):   
    norm = ImageNormalize(data, interval=ZScaleInterval())
    plt.imshow(data, origin='lower', cmap='gray', norm=norm)
    plt.title(title)
    
    if wcs is not None:
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.grid(color='white', ls='dotted')
    
    plt.colorbar(label='Counts')
    plt.show(block=block)
    plt.pause(time)
    plt.close()

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




@dataclass
class Star:
    """Represents a detected star with its centroid and properties."""
    x: float          # Centroid X (sub-pixel)
    y: float          # Centroid Y (sub-pixel)
    brightness: float # Integrated brightness (sum of pixels minus background)
    snr: float        # Signal-to-noise ratio




class AstroImageAligner:
    """
    Full-featured class for aligning astronomical images based on star detection and centroiding.
    This class provides methods for detecting stars, calculating transformations, and aligning 
    images with sub-pixel accuracy.The alignment process is robust to varying numbers of detected 
    stars and can handle different types of transformations (translation, rotation, scaling).    
    """
    
    def __init__(self, 
                 min_brightness_percentile: float = 95.0,
                 min_snr: float = 5.0,
                 max_stars: int = 50,
                 detection_threshold: float = 2.5,
                 centroid_window: int = 5,
                 conf: dict = None):
        """
        Initializes the AstroImageAligner with configurable parameters for star detection and alignment.

        Args:
            min_brightness_percentile (float, optional): Minimum brightness percentile to consider a star 
            for detection. Defaults to 95.0.
            min_snr (float, optional): Minimum signal-to-noise ratio for a detected star to be considered valid.
            Defaults to 5.0.
            max_stars (int, optional): Maximum number of stars to detect and use for alignment. Defaults to 50.
            detection_threshold (float, optional): Threshold in terms of standard deviations above the background 
            for star detection. Defaults to 2.5.
            centroid_window (int, optional): Size of the window (in pixels) around the detected star for centroid 
            refinement. Defaults to 5.
            conf (dict, optional): Configuration dictionary for additional settings. Defaults to None.
        """

        self.min_brightness_percentile = min_brightness_percentile
        self.min_snr = min_snr
        self.max_stars = max_stars
        self.detection_threshold = detection_threshold
        self.centroid_window = centroid_window
        
        # Internal state for reference stars and transformation
        self.reference_stars: Optional[List[Star]] = None
        self.reference_image: Optional[np.ndarray] = None
        self.transformation_matrix: Optional[np.ndarray] = None

        # Additional configuration handling
        self.conf = conf if conf is not None else {}
        self.PATH_REDUCED = Path(self.conf["DIRECTORIES"]["PATH_OUTPUT"])
        self.ic = ccdp.ImageFileCollection(self.PATH_REDUCED, keywords='*',
                                           glob_include='ADP*')
        
        logger.info(f"AstroImageAligner initialized with min_brightness_percentile={self.min_brightness_percentile}, "
                    f"min_snr={self.min_snr}, max_stars={self.max_stars}, detection_threshold={self.detection_threshold}, "
                    f"centroid_window={self.centroid_window}")

    @staticmethod
    def distancia_minima_entre_conjuntos(X, Y, metric='euclidean'):
        """
        Calcula la distancia mínima entre dos conjuntos de puntos.
        
        Parámetros:
        -----------
        X : array-like, shape (n_points_X, n_dimensions)
            Primer conjunto de coordenadas
        Y : array-like, shape (n_points_Y, n_dimensions)
            Segundo conjunto de coordenadas
        metric : str, default='euclidean'
            Métrica de distancia ('euclidean', 'manhattan', 'cosine', etc.)
        
        Retorna:
        --------
        dict : Diccionario con:
            - 'distancia_minima': La distancia más pequeña entre cualquier par de puntos
            - 'par_minimo': Índices (i, j) de los puntos con distancia mínima
            - 'punto_en_X': Coordenada en X con distancia mínima
            - 'punto_en_Y': Coordenada en Y con distancia mínima
            - 'matriz_distancias': Matriz completa de distancias (opcional para debug)
        """
        # Convertir a arrays numpy
        X = np.array(X)
        Y = np.array(Y)
        
        # Asegurar que son 2D (manejar caso de un solo punto)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        # Calcular todas las distancias entre pares (vectorizado y eficiente)
        distancias = cdist(X, Y, metric=metric)
        
        # Encontrar el mínimo
        idx_min = np.unravel_index(np.argmin(distancias), distancias.shape)
        distancia_min = distancias[idx_min]
        
        return {
            'distancia_minima': float(distancia_min),
            'par_minimo': idx_min,
            'punto_en_X': X[idx_min[0]].tolist(),
            'punto_en_Y': Y[idx_min[1]].tolist(),
            'matriz_distancias': distancias
        }
    
    @staticmethod
    def distancia_hausdorff(X, Y):
        """
        Distancia de Hausdorff: el máximo de las distancias mínimas en ambas direcciones.
        Útil para medir qué tan similares son dos formas/conjuntos.
        """
        distancias = cdist(X, Y)
        
        # Distancia mínima de cada punto en X a Y
        d_xy = np.max(np.min(distancias, axis=1))
        # Distancia mínima de cada punto en Y a X  
        d_yx = np.max(np.min(distancias, axis=0))
        
        return max(d_xy, d_yx)


    def find_correspondences(self, src, dst, transform, threshold):
        """
        Encuentra correspondencias uno a uno entre puntos transformados de src y dst.
        
        Parámetros:
            src : ndarray (N,2) - coordenadas originales en imagen de referencia.
            dst : ndarray (M,2) - coordenadas en imagen a alinear.
            transform : objeto transform de skimage que mapea src -> dst.
            threshold : float - distancia máxima para considerar una coincidencia.
        
        Retorna:
            Lista de tuplas (i, j) con índices de las estrellas que corresponden.
        """
        src_trans = transform(src)
        n_src = len(src)
        n_dst = len(dst)
        used_dst = [False] * n_dst
        correspondences = []
        
        for i in range(n_src):
            # distancia a todos los puntos dst
            R = self.distancia_minima_entre_conjuntos(src_trans[i:i+1], dst) #!np.sqrt(np.sum((src_trans[i] - dst) ** 2, axis=1))
            dists = R['matriz_distancias'].flatten()
            # ordenar por distancia ascendente
            order = np.argsort(dists)
            for j in order:
                if not used_dst[j] and dists[j] < threshold:
                    correspondences.append((i, j))
                    used_dst[j] = True
                    break
        return correspondences

    @staticmethod
    def extract_sources(data, kernel, thresh=1.5):
        #?if data.dtype.byteorder != '=' and data.dtype.byteorder != '|':
        #?    data = data.astype(data.dtype.newbyteorder('='))
        sky_background = sep.Background(data)  #Estimate sky background
        data_sub = data - sky_background  #Subtract sky background
        sources = sep.extract(data_sub, thresh=thresh, err=sky_background.globalrms, filter_kernel=kernel)  #Detect sources
        logger.info(f"Detected {len(sources)} sources in the image")
        return sources

    @staticmethod
    def obtain_list_coords_flux(sources):
        coords = np.array([(src['x'], src['y']) for src in sources])
        flux = np.array([src['flux'] for src in sources])
        return coords, flux


    def align_stars(self, coord1, coord2, flux1=None, flux2=None, num_bright=20,
                threshold=2.0, scale_tolerance=0.2):
        """
        Encuentra la transformación de similaridad que alinea las estrellas de coord1 con coord2.
        
        Parámetros:
            coord1, coord2 : arrays de forma (N,2) y (M,2) con las coordenadas (x,y) de las estrellas.
            flux1, flux2   : arrays opcionales con el flujo (brillo) de cada estrella.
                             Si se proporcionan, se usan para seleccionar las más brillantes.
                             Se asume que un valor mayor indica más brillo.
            num_bright     : número de estrellas más brillantes a considerar en la búsqueda inicial.
            threshold      : distancia máxima (en píxeles) para considerar una correspondencia.
            scale_tolerance: tolerancia relativa en la distancia entre pares de estrellas
                             (filtro para descartar pares con escalas muy diferentes).
        
        Retorna:
            transform : objeto transformación (similarity) que mapea coord1 a coord2.
                        Puede ser None si no se encuentra ninguna transformación.
            correspondencias : lista de tuplas (i, j) con los índices de las estrellas que coinciden.
        """
        n1, n2 = len(coord1), len(coord2)
        if n1 == 0 or n2 == 0:
            log_and_raise(ValueError("Cada lista debe contener al menos una estrella."))
        
        # Seleccionar las estrellas más brillantes (si no hay flujo, se toman las primeras num_bright)
        if flux1 is not None:
            # Índices de las num_bright más brillantes (mayor flujo)
            bright_idx1 = np.argsort(flux1)[-num_bright:][::-1]
        else:
            bright_idx1 = np.arange(min(num_bright, n1))
        
        if flux2 is not None:
            bright_idx2 = np.argsort(flux2)[-num_bright:][::-1]
        else:
            bright_idx2 = np.arange(min(num_bright, n2))
        
        src_bright = coord1[bright_idx1]
        dst_bright = coord2[bright_idx2]
        
        # Generar todos los pares de índices dentro de las brillantes
        src_pairs = list(combinations(range(len(src_bright)), 2))
        dst_pairs = list(combinations(range(len(dst_bright)), 2))
        
        # Precalcular distancias de cada par para filtrado rápido
        src_dists = [(i, j, np.linalg.norm(src_bright[i] - src_bright[j])) for i, j in src_pairs]
        dst_dists = [(i, j, np.linalg.norm(dst_bright[i] - dst_bright[j])) for i, j in dst_pairs]
        
        best_score = 0          # mayor número de inliers
        best_transform = None
        best_error = np.inf     # error medio de los inliers (para desempate)
        
        # Búsqueda exhaustiva de la mejor transformación a partir de pares
        for (i1, i2, d_src) in src_dists:
            for (j1, j2, d_dst) in dst_dists:
                # Filtro por escala: la relación de distancias debe estar dentro de la tolerancia
                if abs(d_src - d_dst) / d_src > scale_tolerance:
                    continue
                
                src_pts = src_bright[[i1, i2], :]
                dst_pts = dst_bright[[j1, j2], :]
                #!try:
                tform = estimate_transform('euclidean', src_pts, dst_pts)
                #print(f"Transformación estimada: rotation={tform.rotation:.4f} rad, translation={tform.translation}")
                #!except Exception:
                #!    continue  # transformación degenerada (p.ej. puntos coincidentes)
                
                # Evaluar la transformación con todas las estrellas brillantes
                src_trans = tform(src_bright)
                # Matriz de distancias a las estrellas brillantes de dst
                #!dist_matrix = np.sqrt(((src_trans[:, np.newaxis, :] - dst_bright[np.newaxis, :, :]) ** 2).sum(axis=2))
                #!min_dists = np.min(dist_matrix, axis=1)
                R = self.distancia_minima_entre_conjuntos(src_trans, dst_bright)
                min_dists = R['matriz_distancias'].min(axis=1)
                inliers = np.sum(min_dists < threshold)
                
                # Calcular error medio de los inliers (para desempate)
                if inliers > 0:
                    error_mean = self.distancia_hausdorff(src_trans, dst_bright)
                else:
                    error_mean = np.inf
                
                if (inliers > best_score) or (inliers == best_score and error_mean < best_error):
                    best_score = inliers
                    best_transform = tform
                    best_error = error_mean
        
        # Si no se encontró ninguna transformación con pares, intentar con traslación pura (1 estrella)
        if best_transform is None:
            best_score_ind = 0
            best_transform_ind = None
            for i, pt_src in enumerate(src_bright):
                dists = np.linalg.norm(dst_bright - pt_src, axis=1)
                j = np.argmin(dists)
                if dists[j] < threshold:
                    traslacion = dst_bright[j] - pt_src
                    tform = AffineTransform(translation=traslacion)
                    # Evaluar
                    src_trans = tform(src_bright)
                    #!dist_matrix = np.sqrt(((src_trans[:, np.newaxis, :] - dst_bright[np.newaxis, :, :]) ** 2).sum(axis=2))
                    #!min_dists = np.min(dist_matrix, axis=1)
                    R = self.distancia_minima_entre_conjuntos(src_trans, dst_bright)
                    min_dists = R['matriz_distancias'].min(axis=1)
                    inliers = np.sum(min_dists < threshold)
                    if inliers > best_score_ind:
                        best_score_ind = inliers
                        best_transform_ind = tform
            if best_transform_ind is not None:
                best_transform = best_transform_ind
                best_score = best_score_ind
        
        if best_transform is None:
            # No se pudo determinar ninguna transformación
            return None, []
        
        # Refinamiento usando todas las estrellas (no solo las brillantes)
        self.corr = self.find_correspondences(coord1, coord2, best_transform, threshold)
        
        if len(self.corr) > 2:
            logger.info(f"Refinando transformación con {len(self.corr)} correspondencias.")
            src_pts = coord1[[c[0] for c in self.corr]]
            dst_pts = coord2[[c[1] for c in self.corr]]
            try:
                self.best_transform = estimate_transform('euclidean', src_pts, dst_pts)
            except Exception:
                pass  # si falla, nos quedamos con la transformación anterior
        elif len(self.corr) <= 2 and len(self.corr) > 0:
            if len(self.corr) == 1:
                # Solo una correspondencia: usar traslación pura
                logger.warning("Solo se encontró una correspondencia, usando traslación pura.")
                i, j = self.corr[0]
                traslacion = coord2[j] - coord1[i]
                self.best_transform = AffineTransform(translation=traslacion)
            else:
                logger.warning("Dos correspondencias encontradas, pero no se puede estimar una transformación robusta. Manteniendo la transformación obtenida de las brillantes.")
                i, j = self.corr[0]
                k, l = self.corr[1]
    
                #!traslacion1 = coord2[j] - coord1[i]
                #!k, l = corr[1]
                #!traslacion2 = coord2[l] - coord1[k]
                #!array_translation = np.array([traslacion1, traslacion2])
                #!array_dst = np.array([np.sqrt(traslacion1[0]**2 + traslacion1[1]**2), np.sqrt(traslacion2[0]**2 + traslacion2[1]**2)])
                #!best_transform = AffineTransform(translation=array_translation[np.argmin(array_dst)])
                SRC = coord1[[i, k]]
                DST = coord2[[j, l]]
                self.tform = estimate_transform('euclidean', SRC, DST)
                self.best_transform = AffineTransform(translation=self.tform.translation)
    
        elif (len(self.corr) == 0) and (len(coord1)!=0): #mantener la transformación obtenida de las brillantes (puede que no haya correspondencias)
            # No hay correspondencias, pero hay coordenadas en la imagen original
            # No se actualiza best_transform, se mantiene la transformación obtenida previamente
            pass  # No hacer nada, mantener best_transform como estaba

        return self.best_transform, self.corr

    def run_aligning(self, filt, sky):
        """
        """

        dict_noccd={'Sloan_u':'ccd1', 'Sloan_g': 'ccd2', 'Sloan_r':'ccd3',
                    'Sloan_i': 'ccd4', 'Sloan_z': 'ccd5'}
        
        noccd = dict_noccd[filt]
        
        dict_frames={'FILTRO': f'{filt}', 'SSKY': f'{sky}'}
        aligned_frames = []
        df = pd.DataFrame(columns=['frame', 'num_correspondences', 'transform_type', 
                                   'inlier_error', 'tform'])
        
        pathfiles = self.ic.files_filtered(include_path=True, **dict_frames)
        
        #Load the reference image and its WCS
        ref_data, ref_header = fits.getdata(pathfiles[0], header=True)
        ref_wcs = WCS(ref_header)

        self.ref_header = ref_header.copy() #It's needed out.
        self.ref_wcs = ref_wcs.copy() #It's needed out.
    
        #?ref_data = ref_data.byteswap().view(ref_data.dtype.newbyteorder())
        ref_data = ref_data.astype(ref_data.dtype.newbyteorder('=')) #& ref_data.byteswap().newbyteorder()

        #Create a list to store the coordinates of the detected sources in the reference image
        ref_coords, ref_flux = self.obtain_list_coords_flux(self.extract_sources(ref_data, kernel))

        for path in pathfiles[1:]:
            logger.info(f"Processing frame: {os.path.basename(path)}")   
            #print(f"Processing frame: {path}")
            #Load the example image and its WCS
            img_data, img_header = fits.getdata(path, header=True)
            img_wcs = WCS(img_header)
    
            #?img_data = img_data.byteswap().view(ref_data.dtype.newbyteorder())
            img_data = img_data.astype(img_data.dtype.newbyteorder('=')) #& img_data.byteswap().newbyteorder()
    
            #Show the example image
            if visualize:
                logger.info(f"Processing frame: {path}")
                visualize_image(img_data, f"{path.split('/')[-1]}", img_wcs, time=0.1)
    
            #Create a list to store the coordinates of the detected sources in the example image
            try:
                img_coords, img_flux = self.obtain_list_coords_flux(self.extract_sources(img_data, kernel))
            except Exception as e:
                log_and_raise(ValueError(f"Error extracting sources from {path}: {e}"))
                continue
    
            tform, corr = self.align_stars(img_coords, ref_coords, img_flux, ref_flux, num_bright=20, threshold=4.0)
            logger.info(f"Found {len(corr)} correspondences in frame")
            df = pd.concat([df, pd.DataFrame({'frame': [path.split('/')[-1]], 
                                         'num_correspondences': [len(corr)], 
                                         'transform_type': ['euclidean'], 
                                         'inlier_error': np.nan, #[np.mean([np.linalg.norm(tform(img_coords[c[1]]) - ref_coords[c[0]]) for c in corr]) ], 
                                         'tform': [tform]})], ignore_index=True)
            if len(corr) == 0:
                logger.warning(f"Too few correspondences ({len(corr)}) in frame {path}")
                continue
    
            if tform is not None:
                # Aplicar la transformación a la segunda imagen (o a sus coordenadas)
                # Por ejemplo, para alinear las coordenadas de la segunda imagen a la primera:
                coord2_aligned = tform(img_coords)   # solo si la transformación mapea de coord1 a coord2
                # Nota: La transformación obtenida mapea puntos de coord1 a coord2. 
                # Si quieres transformar la imagen 2 para que coincida con la 1, necesitas la inversa.
                img_aligned = warp(img_data, tform.inverse)
            else:
                log_and_raise(ValueError("No se pudo alinear."))

            #Visualize the aligned image
            if visualize and tform is not None:
                logger.info(f"Visualizing aligned frame: {path.split('/')[-1]} with {len(corr)} correspondences")
                visualize_image(img_aligned, f"ALIGNED_{path.split('/')[-1]} with {len(corr)}", img_wcs, time=1)
            
            aligned_frames.append(img_aligned)
            del tform, corr, img_aligned
            logger.info("Finished processing frame.")

        stack = np.sum(np.array(aligned_frames), axis=0)
        self.num = len(aligned_frames)
    
        if visualize:
            logger.info("Visualizing stacked aligned frames...")
            visualize_image(stack, "STACKED", ref_wcs, block=True)
        
        df.to_csv(self.PATH_REDUCED / f'alignment_results_{noccd}.csv', index=False)
        img = CCDData(stack, header=ref_header, wcs=ref_wcs, unit=u.adu)
        img.write(self.PATH_REDUCED / f'STACKED_{noccd}.fits', overwrite=True)
        self.stacked_image = img





def save_fits(image, header, wcs, fname, allow_nosky=True):
    """
    This method saves the stacked image.

    Args:
        image (float): Data to save.
        header (str): Header for the stacked image.
        wcs (str): WCS information for the stacked image.
        fname (str): Name for the stacked image.
        allow_nosky (bool, optional): Whether to allow saving images with 'NOSKY' in the filename. Defaults to True.
    
    Raises:
        ValueError: If the filename contains 'NOSKY' and allow_nosky is False.
    """
    if 'NOSKY' in fname and not allow_nosky:
        log_and_raise(ValueError("Saving images with 'NOSKY' in the filename is not allowed. Please check the filename and try again."))
    header['STACKED'] = 'YES'
    ccd = CCDData(data=image, header=header, wcs=wcs, unit='adu')
    ccd.write(fname, overwrite=True)
    logger.info(f"New image has been created: {os.path.basename(fname)}")