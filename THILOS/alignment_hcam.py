
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
        Calculate the minimum distance between two sets of points.
        
        Parameters:
        -----------
        X : array-like, shape (n_points_X, n_dimensions)
            First set of coordinates
        Y : array-like, shape (n_points_Y, n_dimensions)
            Second set of coordinates
        metric : str, default='euclidean'
            Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
        
        Returns:
        --------
        dict : Dictionary with:
            - 'distancia_minima': The smallest distance between any pair of points
            - 'par_minimo': Indices (i, j) of the points with minimum distance
            - 'punto_en_X': Coordinate in X with minimum distance
            - 'punto_en_Y': Coordinate in Y with minimum distance
            - 'matriz_distancias': Complete distance matrix (optional for debug)
        """
        # Convert to arrays
        X = np.array(X)
        Y = np.array(Y)
        
        # Check if arrays are 1D and reshape if necessary
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        # Calculate all the distances between pairs (vectorized and efficient)
        distancias = cdist(X, Y, metric=metric)
        
        # Find the minimum distance and the corresponding indices
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
        Hausdorff distance: the maximum of the minimum distances in both directions.
        Useful for measuring how similar two shapes/sets are.
        """
        distancias = cdist(X, Y)
        
        # Minimum distance of each point in X to Y
        d_xy = np.max(np.min(distancias, axis=1))
        # Minimum distance of each point in Y to X
        d_yx = np.max(np.min(distancias, axis=0))
        
        return max(d_xy, d_yx)


    def find_correspondences(self, src, dst, transform, threshold):
        """
        Finds one-to-one correspondences between transformed points from src and dst.
        
        Parameters:
            src : ndarray (N,2) - original coordinates in reference image.
            dst : ndarray (M,2) - coordinates in image to align.
            transform : skimage transform object that maps src -> dst.
            threshold : float - maximum distance to consider a match.
        
        Returns:
            List of tuples (i, j) with indices of the corresponding stars.
        """
        src_trans = transform(src)
        n_src = len(src)
        n_dst = len(dst)
        used_dst = [False] * n_dst
        correspondences = []
        
        for i in range(n_src):
            # distance to all the points dst
            R = self.distancia_minima_entre_conjuntos(src_trans[i:i+1], dst)
            dists = R['matriz_distancias'].flatten()
            # sort by distance ascending
            order = np.argsort(dists)
            for j in order:
                if not used_dst[j] and dists[j] < threshold:
                    correspondences.append((i, j))
                    used_dst[j] = True
                    break
        return correspondences

    @staticmethod
    def extract_sources(data, kernel, thresh=1.5):
        """This function 

        Args:
            data (_type_): _data_ from which to extract sources (2D array).
            kernel (_type_): Kernel to use for source detection (e.g., a Gaussian kernel).
            thresh (float, optional): Threshold in terms of standard deviations above the background 
            for source detection. Defaults to 1.5.

        Returns:
            tab : List of detected sources with their properties (e.g., x, y, flux).
        """
        sky_background = sep.Background(data)  #Estimate sky background
        data_sub = data - sky_background  #Subtract sky background
        sources = sep.extract(data_sub, thresh=thresh, err=sky_background.globalrms, filter_kernel=kernel)  #Detect sources
        logger.info(f"Detected {len(sources)} sources in the image")
        return sources

    @staticmethod
    def obtain_list_coords_flux(sources):
        """This function obtains the coordinates and flux of the detected sources.

        Args:
            sources (list): List of detected sources with their properties.

        Returns:
            tuple: Arrays of coordinates and flux values.
        """

        coords = np.array([(src['x'], src['y']) for src in sources])
        flux = np.array([src['flux'] for src in sources])
        return coords, flux


    def align_stars(self, coord1, coord2, flux1=None, flux2=None, num_bright=20,
                threshold=2.0, scale_tolerance=0.2):
        """
        Finds the similarity transformation that aligns the stars from coord1 with coord2.
        
        Parameters:
            coord1, coord2 : arrays of shape (N,2) and (M,2) with the coordinates (x,y) of the stars.
            flux1, flux2   : optional arrays with the flux (brightness) of each star.
                             If provided, they are used to select the brightest stars.
                             It is assumed that a higher value indicates more brightness.
            num_bright     : number of brightest stars to consider in the initial search.
            threshold      : maximum distance (in pixels) to consider a correspondence.
            scale_tolerance: relative tolerance in the distance between pairs of stars
                             (filter to discard pairs with very different scales).
        
        Returns:
            transform : similarity transformation object that maps coord1 to coord2.
                        Can be None if no transformation is found.
            correspondencias : list of tuples (i, j) with the indices of the stars that match.
        """
        n1, n2 = len(coord1), len(coord2)
        if n1 == 0 or n2 == 0:
            log_and_raise(ValueError("Each list must contain at least one star."))
        
        # Select the brightest stars (if no flux is provided, the first num_bright are taken)
        if flux1 is not None:
            # Indices of the num_bright brightest stars (highest flux)
            bright_idx1 = np.argsort(flux1)[-num_bright:][::-1]
        else:
            bright_idx1 = np.arange(min(num_bright, n1))
        
        if flux2 is not None:
            bright_idx2 = np.argsort(flux2)[-num_bright:][::-1]
        else:
            bright_idx2 = np.arange(min(num_bright, n2))
        
        src_bright = coord1[bright_idx1]
        dst_bright = coord2[bright_idx2]
        
        # Generate all pairs of indices within the brightest
        src_pairs = list(combinations(range(len(src_bright)), 2))
        dst_pairs = list(combinations(range(len(dst_bright)), 2))
        
        # Pre-calculate distances of each pair for fast filtering
        src_dists = [(i, j, np.linalg.norm(src_bright[i] - src_bright[j])) for i, j in src_pairs]
        dst_dists = [(i, j, np.linalg.norm(dst_bright[i] - dst_bright[j])) for i, j in dst_pairs]
        
        best_score = 0          # the number of inliers
        best_transform = None
        best_error = np.inf     # average error of the inliers (for tie-breaking)
        
        # Exhaustive search for the best transformation from pairs
        for (i1, i2, d_src) in src_dists:
            for (j1, j2, d_dst) in dst_dists:
                # Filter by scale: the ratio of distances must be within the tolerance
                if abs(d_src - d_dst) / d_src > scale_tolerance:
                    continue
                
                src_pts = src_bright[[i1, i2], :]
                dst_pts = dst_bright[[j1, j2], :]

                tform = estimate_transform('euclidean', src_pts, dst_pts)
                
                # Evaluate the transformation with all the bright stars
                src_trans = tform(src_bright)
                # Distance matrix to the bright stars in dst
                R = self.distancia_minima_entre_conjuntos(src_trans, dst_bright)
                min_dists = R['matriz_distancias'].min(axis=1)
                inliers = np.sum(min_dists < threshold)
                
                # Calculate average error of the inliers (for tie-breaking)
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
                    # Evaluate the transformation with all the bright stars
                    src_trans = tform(src_bright)
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
            # It means that no transformation was found, even with the single-star translation.
            logger.warning("No finding transformation with the brightest stars. Returning None and empty correspondences.")
            return None, []
        
        # Refine the transformation using all the stars (not just the bright ones)
        self.corr = self.find_correspondences(coord1, coord2, best_transform, threshold)
        
        if len(self.corr) > 2:
            logger.info(f"Refining transformation with {len(self.corr)} correspondences.")
            src_pts = coord1[[c[0] for c in self.corr]]
            dst_pts = coord2[[c[1] for c in self.corr]]
            try:
                self.best_transform = estimate_transform('euclidean', src_pts, dst_pts)
            except Exception:
                pass  # if it fails, we stay with the previous transformation
        elif len(self.corr) <= 2 and len(self.corr) > 0:
            if len(self.corr) == 1:
                # Only one correspondence, we can only do a pure translation
                logger.warning("Only one correspondence found, using pure translation.")
                i, j = self.corr[0]
                traslacion = coord2[j] - coord1[i]
                self.best_transform = AffineTransform(translation=traslacion)
            else:
                logger.warning("Two correspondences found, but no robust transformation can be estimated. Keeping the transformation obtained from the brightest stars.")
                i, j = self.corr[0]
                k, l = self.corr[1]
                SRC = coord1[[i, k]]
                DST = coord2[[j, l]]
                self.tform = estimate_transform('euclidean', SRC, DST)
                self.best_transform = AffineTransform(translation=self.tform.translation)
    
        elif (len(self.corr) == 0) and (len(coord1)!=0): #Keep the transformation obtained from the brightest stars (there might not be correspondences)
            # There are no correspondences, but there are coordinates in the original image, so we keep the transformation obtained from the brightest stars (if any)
            logger.warning("No correspondences found, but there are stars in the original image. Keeping the transformation obtained from the brightest stars (if any).")
            # No update best_transform, the transformation obtained previously from the brightest stars is kept, even if it has no correspondences. This can happen if the threshold is too strict or if the images are very different, but we still want to have a transformation to try to align the images (even if it might not be perfect).
            pass  # No update to best_transform, we keep the one obtained from the brightest stars (even if it has no correspondences)

        else:
            logger.warning("No correspondences found and no stars in the original image. Returning None and empty correspondences.")
            return None, []

        return self.best_transform, self.corr

    def run_aligning(self, filt, sky):
        """This method runs the alignment process for a given filter and sky condition.

        Args:
            filt (str): Filter name to select the frames for alignment.
            sky (str): Sky condition to select the frames for alignment.
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
    
        ref_data = ref_data.astype(ref_data.dtype.newbyteorder('='))

        #Create a list to store the coordinates of the detected sources in the reference image
        ref_coords, ref_flux = self.obtain_list_coords_flux(self.extract_sources(ref_data, kernel))

        for path in pathfiles[1:]:
            logger.info(f"Processing frame: {os.path.basename(path)}")   
            img_data, img_header = fits.getdata(path, header=True)
            img_wcs = WCS(img_header)
    
            img_data = img_data.astype(img_data.dtype.newbyteorder('='))
    
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
                                         'inlier_error': np.nan, 
                                         'tform': [tform]})], ignore_index=True)
            if len(corr) == 0:
                logger.warning(f"Too few correspondences ({len(corr)}) in frame {path}")
                continue
    
            if tform is not None:
                # Apply the transformation to the second image (or its coordinates)
                # For example, to align the coordinates of the second image to the first:
                coord2_aligned = tform(img_coords)   # only if the transformation maps from coord1 to coord2
                # Note: The obtained transformation maps points from coord1 to coord2.
                # If you want to transform image 2 so it matches image 1, you need the inverse.
                img_aligned = warp(img_data, tform.inverse)
            else:
                log_and_raise(ValueError("It could not align."))

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