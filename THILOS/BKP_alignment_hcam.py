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

from pathlib import Path
import numpy as np
import cv2, os
from scipy import ndimage
from scipy.optimize import minimize
import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.table import Table
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from loguru import logger
import warnings

from Color_Codes import bcolors as bcl

import logging, inspect

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




    def detect_stars(self, image: np.ndarray) -> List[Star]:
        """
        Detect stars in the given image using a robust centroiding algorithm.
        
        Algorithm steps:
        1. Estimate background and noise using robust statistics (median and MAD).
        2. Define detection threshold as background + N*sigma.
        3. Find connected regions of pixels above the threshold.
        4. For each region, calculate the weighted centroid and brightness.
        5. Filter stars by SNR and sort by brightness.
        6. Return the top N stars as a list of Star objects.
        
        This method is designed to be robust against varying background levels, noise, and 
        can handle both faint and bright stars effectively. It also provides sub-pixel accuracy 
        in the centroiding process, which is crucial for precise image alignment.
        
        Args:
            image (np.ndarray): The input image in which to detect stars. Can be a 2D grayscale 
            image or a 3D color image (in which case it will be converted to grayscale).
            
        Returns:
            List of Star objects sorted by brightness in descending order, limited to the maximum 
            number of stars specified by `self.max_stars`.
        """
        # Convert to grayscale if it's a color image (simple average across channels)
        if len(image.shape) == 3:
            img_gray = np.mean(image, axis=2)
            logger.debug("Input image is color, converted to grayscale for star detection.")
        else:
            img_gray = image.astype(np.float64)
            logger.debug("Input image is already grayscale, proceeding with star detection.")
            
        # Normalize the image to [0, 1] for better numerical stability in background and noise estimation
        img_norm = img_gray / img_gray.max() if img_gray.max() > 0 else img_gray
        logger.debug("Image normalized for star detection.")
        
        # 1. Estimate background and noise using robust statistics
        background = np.median(img_norm)
        mad = np.median(np.abs(img_norm - background))
        noise = 1.4826 * mad  # Estimator of standard deviation from MAD
        logger.debug(f"Estimated background: {background:.4f}, noise (sigma): {noise:.4f}")
        
        if noise < 1e-10:
            noise = np.std(img_norm) * 0.1
            logger.warning(f"Noise estimation from MAD is very low. Using 10% of the standard deviation: {noise:.4f}")
            
        # 2. Threshold for star detection
        threshold = background + self.detection_threshold * noise
        logger.debug(f"Detection threshold set at: {threshold:.4f}")

        
        # 3. Find connected regions of pixels above the threshold
        binary_mask = img_norm > threshold
        labeled, num_features = ndimage.label(binary_mask)
        logger.info(f"Detected {num_features} initial star candidates above the threshold.")
        
        stars = []
        
        for i in range(1, num_features + 1):
            # Coordinates of pixels in the current region
            y_coords, x_coords = np.where(labeled == i)
            
            if len(y_coords) < 3:  # Reject very small regions (likely noise)
                continue
                
            # Extract pixel values for the current region
            pixel_values = img_norm[y_coords, x_coords]
            
            # Calculate brightness and weights for centroiding
            weights = pixel_values - background
            weights = np.maximum(weights, 0)
            
            if weights.sum() < 1e-10:
                continue
                
            # 4. Centroid calculation (weighted average)
            centroid_x = np.sum(x_coords * weights) / weights.sum()
            centroid_y = np.sum(y_coords * weights) / weights.sum()
            
            # Refinement of centroid using a small window around the initial estimate could be implemented here
            # (omitting for simplicity, but could be added for improved accuracy)
            
            # Calculate SNR
            signal = np.sum(weights)
            noise_total = noise * np.sqrt(len(pixel_values))
            snr = signal / noise_total if noise_total > 0 else 0
            
            if snr >= self.min_snr:
                stars.append(Star(
                    x=float(centroid_x),
                    y=float(centroid_y),
                    brightness=float(signal),
                    snr=float(snr)
                ))
            
            logger.debug(f"Star candidate {i}: Centroid=({centroid_x:.2f}, {centroid_y:.2f}), Brightness={signal:.2f}, SNR={snr:.2f}")
        
        # 5. Sort stars by brightness and limit to max_stars
        stars.sort(key=lambda s: s.brightness, reverse=True)
        return stars[:self.max_stars]




    def find_transformation(self, 
                          stars_target: List[Star], 
                          stars_reference: List[Star],
                          method: str = 'rigid') -> np.ndarray:
        """
        Calculate the transformation matrix to align the target image to the reference image based on detected stars.
        
        Strategy for determining the type of transformation:
        - 1 star: Only translation can be determined (no rotation or scaling).
        - 2 stars: Can determine translation and rotation (rigid transformation), but not scaling.
        - 3 or more stars: Can determine translation, rotation, and scaling (affine transformation).
        
        Args:
            stars_target (List[Star]): List of detected stars in the target image.
            stars_reference (List[Star]): List of detected stars in the reference image.
            method (str, optional): Method to use for transformation. Options are 'auto', 'translation', 
            'rigid', 'affine'. Defaults to 'auto', which determines the method based on the number of stars.
            
        Returns:
            Matrix (3x3) representing the transformation to align the target image to the reference image. 
            This matrix can be used with cv2.warpPerspective for alignment.
        
        Raises:
            ValueError: If there are not enough stars to compute the transformation or if an unknown method is specified.
        """
        n_target = len(stars_target)
        n_ref = len(stars_reference)
        
        if n_target == 0 or n_ref == 0:
            log_and_raise(ValueError("Stars not detected in one or both images. Cannot compute transformation."))
        
        # Determination of transformation method based on the number of detected stars
        if method == 'auto':
            if n_target >= 3 and n_ref >= 3:
                method = 'affine'
            elif n_target >= 2 and n_ref >= 2:
                method = 'rigid'
            else:
                method = 'translation'
        
        logger.info(f"Finding transformation using method: {method} (Target stars: {n_target}, Reference stars: {n_ref})")
        
        # Case 1: Only translation (1 star)
        if method == 'translation':
            # Use the first star in each list for translation (assuming they are the same star)
            ref = stars_reference[0]
            tgt = stars_target[0]
            
            dx = ref.x - tgt.x
            dy = ref.y - tgt.y
            
            M = np.array([
                [1, 0, dx],
                [0, 1, dy],
                [0, 0, 1]
            ], dtype=np.float64)
            return M
        
        # Case 2: Rigid transformation (translation + rotation, no scaling)
        elif method == 'rigid':
            return self._find_rigid_transform(stars_target, stars_reference)
        
        # Case 3: Affine transformation (translation + rotation + scaling)
        elif method == 'affine':
            return self._find_affine_transform(stars_target, stars_reference)
        
        else:
            log_and_raise(ValueError(f"Unknown method: {method}"))




    def _match_stars(self, 
                     stars_target: List[Star], 
                     stars_reference: List[Star]) -> List[Tuple[int, int]]:
        """
        Pair stars in the target and reference lists based on proximity and brightness similarity.
        
        For few stars (<5): use a simple nearest neighbor approach with brightness weighting.
        For many stars: could implement a more robust matching (e.g., RANSAC) to handle outliers, 
        but for simplicity.

        This method assumes that the images are already approximately aligned (e.g., from a previous 
        step) and that the stars are in roughly the same order.

        Args:
            stars_target (List[Star]): List of detected stars in the target image.
            stars_reference (List[Star]): List of detected stars in the reference image.

        Returns:
            List of tuples (i, j) where i is the index of the star in the target list and j is the index 
            of the matched star in the reference list.
        
        Raises:
            ValueError: If there are not enough stars to perform matching.
        """
        if len(stars_target) == 1 and len(stars_reference) == 1:
            return [(0, 0)]
        
        # Algorithm for matching stars based on proximity and brightness similarity
        matches = []
        used_ref = set()
        
        for i, tgt in enumerate(stars_target):
            best_match = None
            best_score = float('inf')
            
            for j, ref in enumerate(stars_reference):
                if j in used_ref:
                    continue
                
                # Euclidean distance between centroids
                dist = np.hypot(tgt.x - ref.x, tgt.y - ref.y)
                
                # Weighted score combining distance and brightness difference (normalized)
                brightness_diff = abs(tgt.brightness - ref.brightness) / max(ref.brightness, 1e-10)
                
                # Combine distance and brightness difference into a single score (weights can be tuned)
                score = dist + brightness_diff * 10
                
                if score < best_score:
                    best_score = score
                    best_match = j
            
            if best_match is not None:
                matches.append((i, best_match))
                used_ref.add(best_match)
            
            logger.debug(f"Star {i} in target matched with star {best_match} in reference (score: {best_score:.4f})")
        
        return matches




    def _find_rigid_transform(self, 
                              stars_target: List[Star], 
                              stars_reference: List[Star]) -> np.ndarray:
        """
        Find rigid transformation (translation + rotation) using the Kabsch algorithm for point cloud alignment.
        
        Args:
            stars_target (List[Star]): List of detected stars in the target image.
            stars_reference (List[Star]): List of detected stars in the reference image.

        Returns:
            Matrix (3x3) representing the rigid transformation to align the target image to the reference image.
        """
        matches = self._match_stars(stars_target, stars_reference)
        logger.info(f"Number of matched stars for rigid transformation: {len(matches)}")
        
        if len(matches) < 2:
            log_and_raise(ValueError("At least 2 matched stars are required for rigid transformation."))
            return self.find_transformation(stars_target, stars_reference, method='translation')
        
        # Get the matched points
        tgt_points = np.array([[stars_target[i].x, stars_target[i].y] for i, _ in matches])
        ref_points = np.array([[stars_reference[j].x, stars_reference[j].y] for _, j in matches])
        
        # Center the points
        tgt_center = np.mean(tgt_points, axis=0)
        ref_center = np.mean(ref_points, axis=0)
        
        tgt_centered = tgt_points - tgt_center
        ref_centered = ref_points - ref_center

        
        # Compute covariance matrix
        logger.debug("Computing covariance matrix for rigid transformation.")
        H = tgt_centered.T @ ref_centered
        logger.debug(f"Covariance matrix H:\n{H}")
        
        # Perform SVD to find optimal rotation
        logger.debug("Performing SVD for rotation estimation.")
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        logger.debug(f"Estimated rotation matrix R:\n{R}")
        
        # Improve rotation estimation by ensuring a proper rotation (no reflection)
        logger.debug("Checking for reflection in the rotation matrix.")
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        logger.debug(f"Final rotation matrix R after reflection check:\n{R}")
        
        # Estimate translation
        t = ref_center - R @ tgt_center
        logger.debug(f"Estimated translation vector t: {t}")
        
        # Build the full transformation matrix (3x3)
        M = np.eye(3, dtype=np.float64)
        M[:2, :2] = R
        M[:2, 2] = t
        
        return M




    def _find_affine_transform(self, 
                               stars_target: List[Star], 
                               stars_reference: List[Star]) -> np.ndarray:
        """
        Find affine transformation (translation + rotation + scaling) using matched star points and RANSAC for robustness.
        
        Args:
            stars_target (List[Star]): List of detected stars in the target image.
            stars_reference (List[Star]): List of detected stars in the reference image.
        Returns:
            Matrix (3x3) representing the affine transformation to align the target image to the reference image.

        Raises:
            ValueError: If there are not enough matched stars to compute the affine transformation.
        """
        matches = self._match_stars(stars_target, stars_reference)
        logger.info(f"Number of matched stars for affine transformation: {len(matches)}")
        
        if len(matches) < 3:
            logger.warning("Less than 3 matched stars. Falling back to rigid transformation.")
            logger.debug(f"Stars in target: {stars_target}")
            logger.debug(f"Stars in reference: {stars_reference}")
            logger.debug(f"Matches found: {matches}")
            logger.debug("Using rigid transformation instead.")
            return self._find_rigid_transform(stars_target, stars_reference)
        
        tgt_points = np.array([[stars_target[i].x, stars_target[i].y] for i, _ in matches])
        ref_points = np.array([[stars_reference[j].x, stars_reference[j].y] for _, j in matches])
        
        # Usar OpenCV para estimación robusta
        logger.debug("Estimating affine transformation with RANSAC.")
        M_cv, inliers = cv2.estimateAffinePartial2D(
            tgt_points.astype(np.float32),
            ref_points.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000
        )

        if M_cv is None:
            log_and_raise(ValueError("Affine transformation estimation failed. Not enough inliers or degenerate configuration."))
        
        # Convertir a 3x3
        M = np.eye(3, dtype=np.float64)
        M[:2, :] = M_cv

        logger.debug(f"Estimated affine transformation matrix M:\n{M}")
        
        return M




    def align_image(self, 
                   image: np.ndarray, 
                   reference_image: Optional[np.ndarray] = None,
                   return_stars: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[Star], List[Star]]]:
        """
        Align the given image to the reference image using detected stars and the calculated transformation.
        
        Args:
            image (np.ndarray): The target image to be aligned.
            reference_image (Optional[np.ndarray]): An optional reference image. If provided, it will be set 
            as the new reference for future alignments. If not provided, the existing reference will be used.
            return_stars (bool): If True, returns a tuple containing the aligned image, the list of detected 
            stars in the target image, and the list of reference stars. If False, returns only the aligned image.
        
        Returns:
            If return_stars is False: The aligned image as a numpy array.
            If return_stars is True: A tuple (aligned_image, stars_target, reference_stars) where aligned_image 
            is the aligned image, stars_target is the list of detected stars in the
            target image, and reference_stars is the list of stars in the reference image.

        Raises:
            ValueError: If there are no stars detected in the target image or if there is no reference image set 
            when needed for alignment.
        """
        # Detect stars in the target image
        stars_target = self.detect_stars(image)

        logger.info(f"Detected {len(stars_target)} stars in target image for alignment.")
        
        if len(stars_target) == 0:
            log_and_raise(ValueError("Stars not detected in target image. Cannot align."))
            if return_stars:
                return image, [], []
            return image
        
        # If a new reference image is provided, set it and detect its stars
        if reference_image is not None:
            self.set_reference(reference_image)
        
        if self.reference_stars is None:
            log_and_raise(ValueError("No reference image with detected stars has been set. Cannot align."))
        
        # Calculate transformation matrix
        M = self.find_transformation(stars_target, self.reference_stars)
        self.transformation_matrix = M
        
        # Apply transformation to align the image
        aligned = self._apply_transformation(image, M)

        logger.info("Image aligned successfully.")
        
        if return_stars:
            return aligned, stars_target, self.reference_stars
        return aligned




    def _apply_transformation(self, 
                              image: np.ndarray, 
                              M: np.ndarray) -> np.ndarray:
        """
        Apply the matrix transformation to the image using Lanczos interpolation for high-quality resampling.
        
        Args:
            image (np.ndarray): The input image to be transformed.
            M (np.ndarray): The 3x3 transformation matrix to be applied to the image.
        
        Returns:
            The transformed image as a numpy array.
        """
        h, w = image.shape[:2]
        
        # Keep the same data type and number of channels as the input image
        if len(image.shape) == 3:
            # Process each channel separately to preserve data type and avoid issues with multi-channel warping
            aligned_channels = []
            for c in range(image.shape[2]):
                ch = cv2.warpPerspective(
                    image[:, :, c], M, (w, h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                aligned_channels.append(ch)
            aligned = np.stack(aligned_channels, axis=2)
            logger.debug(f"Aligned image shape: {aligned.shape}, dtype: {aligned.dtype}")
        else:
            aligned = cv2.warpPerspective(
                image, M, (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            logger.debug(f"Aligned image shape: {aligned.shape}, dtype: {aligned.dtype}")
        
        return aligned




    def set_reference(self, image: np.ndarray):
        """
        Establish the reference image and detect its stars for future alignments. This method should be called 
        before aligning any images to set the baseline for alignment.
        
        Detect and store the stars in the reference image, which will be used for calculating transformations 
        when aligning target images.

        Args:
            image (np.ndarray): The image to be set as the reference for alignment. Stars will be detected in 
            this image and stored for future use in alignment calculations.
        
        Raises:
            ValueError: If no stars are detected in the reference image, a warning is issued but the reference 
            is still set (alignment will be limited to translation if only 1 star is detected).
        """
        self.reference_image = image.copy()
        self.reference_stars = self.detect_stars(image)
        
        logger.info(f"Referencia establecida: {len(self.reference_stars)} estrellas detectadas")
        
        if len(self.reference_stars) == 0:
            log_and_raise(ValueError("No stars detected in reference image. Alignment will not be possible."))
        elif len(self.reference_stars) == 1:
            logger.warning("Only 1 star detected in reference image. Alignment will be limited to translation.")
        elif len(self.reference_stars) == 2:
            logger.warning("Only 2 stars detected in reference image. Alignment will be limited to rigid transformation (translation + rotation).")
        else:
            logger.info(f"{len(self.reference_stars)} stars detected in reference image. Full affine transformation will be possible for alignment.")




    def stack_images(self, 
                    images: List[np.ndarray],
                    method: str = 'mean',
                    rejection: str = 'none') -> np.ndarray:
        """
        Align and stack a list of images using the specified method and pixel rejection strategy.
        
        Args:
            images (List[np.ndarray]): List of images to be aligned and stacked. The first image 
            in the list will be used as the reference for alignment.
            method (str, optional): Method to combine the aligned images. Options are 'mean', 'median',
            'sum', 'sigma_clip'. Defaults to 'mean'.
            rejection (str, optional): Pixel rejection strategy to apply during stacking. Options are 
            'none', 'minmax'. Defaults to 'none'.
            
        Returns:
            The final stacked image as a numpy array after alignment and combination.
        
        Raises:
            ValueError: If an unknown stacking method or rejection strategy is specified.
        """
        if len(images) < 2:
            return images[0] if images else None
        
        # Establish the first image as the reference for alignment
        self.set_reference(images[0])
        logger.info(f"Stacking {len(images)} images using method: {method} with rejection: {rejection}")
        
        # Align each image to the reference and store in a list
        aligned_images = [images[0]]  # The reference image is the first one
        
        for i, img in enumerate(images[1:], 1):
            logger.info(f"Aligning image {i+1}/{len(images)}")
            aligned = self.align_image(img)
            aligned_images.append(aligned)
            self.num = i + 1
        
        # Convert to stack 3D (N, H, W) o (N, H, W, C)
        stack = np.array(aligned_images)

        # Apply stacking method
        if method == 'mean':
            result = np.mean(stack, axis=0)
        elif method == 'median':
            result = np.median(stack, axis=0)
        elif method == 'sum':
            result = np.sum(stack, axis=0)
        elif method == 'sigma_clip':
            result = self._sigma_clip_combine(stack, low=3.0, high=3.0)
        else:
            log_and_raise(ValueError(f"Unknown stacking method: {method}"))
        
        # Apply pixel rejection if requested
        if rejection == 'minmax':
            # Each pixel, sort the values across the stack and remove the min and max before averaging
            logger.debug("Applying min-max pixel rejection.")
            sorted_stack = np.sort(stack, axis=0)
            clipped = sorted_stack[1:-1]  # Remove min and max
            result = np.mean(clipped, axis=0)
            logger.debug("Min-max pixel rejection applied.")
        elif rejection != 'none':
            log_and_raise(ValueError(f"Unknown rejection method: {rejection}"))
        
        return result




    def _sigma_clip_combine(self, 
                           stack: np.ndarray, 
                           low: float = 3.0, 
                           high: float = 3.0,
                           max_iter: int = 3) -> np.ndarray:
        """
        Combining images using sigma clipping to reject outliers. This method iteratively calculates the median and
        the median absolute deviation (MAD) to identify and exclude outlier pixels that deviate significantly from 
        the median. The final combined image is computed using the remaining valid pixels after clipping.

        Args:
            stack (np.ndarray): A 3D array of shape (N, H, W) or (N, H, W, C) containing the aligned images to be combined.
            low (float, optional): The lower sigma threshold for clipping. Pixels that are more than `low` sigma 
            below the median will be rejected. Defaults to 3.0.
            high (float, optional): The upper sigma threshold for clipping. Pixels that are more than `high` sigma 
            above the median will be rejected. Defaults to 3.0.
            max_iter (int, optional): The maximum number of iterations for the sigma clipping process. Defaults to 3.

        Returns:
            np.ndarray: The combined image after applying sigma clipping to reject outliers. The shape of the output 
            will be (H, W) for grayscale images or (H, W, C)
        """
        # Calculate initial median and MAD for the stack
        median = np.median(stack, axis=0)
        
        for iteration in range(max_iter):
            # Desviation from the median
            diff = stack - median
            
            # Estimate sigma using MAD
            mad = np.median(np.abs(diff), axis=0)
            sigma = 1.4826 * mad
            
            # Avoid division by zero in case of very low noise
            sigma = np.maximum(sigma, 1e-10)
            logger.debug(f"Iteration {iteration+1}: sigma estimated with MAD, max sigma value: {np.max(sigma):.4f}")
            
            # Mask of valid pixels within the low and high sigma thresholds
            mask = (diff > -low * sigma) & (diff < high * sigma)
            
            # Recompute the median using only the valid pixels
            masked_sum = np.sum(stack * mask, axis=0)
            masked_count = np.sum(mask, axis=0)
            
            # Avoid division by zero when calculating the new median
            masked_count = np.maximum(masked_count, 1)
            new_median = masked_sum / masked_count
            
            # Update the median for the next iteration
            median = new_median
            logger.debug(f"Iteration {iteration+1}: median updated after sigma clipping. The value range of the new median is [{np.min(median):.4f}, {np.max(median):.4f}]")
        
        return median




    def _load_frames(self, filt: str, sky: str) -> List[str]:
        """
        This method retrieves a list of science frames for each filter.

        Args:
            filt (str): Filter name
            sky (str): Sky subtraction (e.g., 'SKY', 'NOSKY', etc.)

        Returns:
            list: List of paths to science frames for the specified filter and sky condition.
        """
        self.tab = Table(self.ic.summary)
        sub_tab = self.tab[(self.tab['filtro']==filt) & (self.tab['ssky']==sky)]
        logger.info(f"Looking for frames that have {filt} and {sky}")
        logger.info(f"{sub_tab}")
        self.total_exptime = sub_tab['exptime'].value.data[0]
        logger.info(f"Exposure time per frame: {self.total_exptime} sec")
        return self.ic.files_filtered(imgtype="SCIENCE",
                                      filtro=filt,
                                      ssky = sky,
                                      include_path=True)




    def _get_each_data(self, filt: str, sky: str) -> List[np.ndarray]:
        """
        This methods reads a list containing the paths to several frames 
        and opens them to add them to a new list.

        Args:
            filt (str): Filter name
            sky (str): Sky subtraction (e.g., 'SKY', 'NOSKY', etc.)

        Returns:
            list: List of science frames for each filter (matrices)
        """
        ccd = []
        for frame_path in self._load_frames(filt, sky=sky):
            ccd.append(CCDData.read(frame_path, unit='adu', hdu=0).data)

        return ccd




    def aligning(self, filt: str, sky: str='SKY') -> np.ndarray:
        """
        This method aligns the science frames taken with the same filter.

        Args:
            filt (str): Filter name
            sky (str): Sky subtraction (e.g., 'SKY', 'NOSKY', etc.). Defaults to 'SKY'.

        Returns:
            float: Stacked image obtained by combining multiple science frames.
        """
        logger.info(f"Creating cube with frames for {filt}")
        cube = self._get_each_data(filt, sky=sky)
        REF = cube[0].astype('float32')
        self.set_reference(REF)
        cube = [IMG.astype('float32') for IMG in cube[1:]]
        return self.stack_images(cube, method=self.conf["ALIGNING"]["stacking_method"])




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