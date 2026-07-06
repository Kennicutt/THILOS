"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2026 Gran Telescopio Canarias <https://www.gtc.iac.es>
Fabricio Manuel Pérez Toledo <fabricio.perez@gtc.iac.es>
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import block_reduce
from typing import Callable, Tuple, Union
from loguru import logger
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


def _check_factor(factor: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Validates and normalizes the binning factor for the rebinning process.

    This function checks if the provided factor is either a positive integer or a tuple of two positive integers. 
    If the factor is a single integer, it is applied to both dimensions (Y and X). If the factor is a tuple, 
    it must contain exactly two positive integers, which will be used for the Y and X dimensions respectively.

    Args:
        factor (Union[int, Tuple[int, int]]): The binning factor to be validated. 
        It can be either a single positive integer or a tuple of two positive integers.
    
    Returns:
        Tuple[int, int]: A tuple containing the normalized binning factors for the Y and X dimensions.
    
    Raises:
        ValueError: If the factor is not a positive integer or a tuple of two positive integers.
    """
    if isinstance(factor, int):
        if factor <= 0:
            log_and_raise(ValueError("Factor must be a positive integer."))
        return (factor, factor)
    elif (isinstance(factor, tuple) and len(factor) == 2
          and all(isinstance(f, int) and f > 0 for f in factor)):
        logger.debug(f"Using different factors for Y and X: {factor}")
        return factor
    else:
        log_and_raise(ValueError("Factor must be an integer or a tuple of two integers."))
    
def _update_wcs(wcs: WCS, factor: Tuple[int, int]) -> WCS:
    """
    Updates the World Coordinate System (WCS) information in the FITS header after rebinning.
    This function takes the original WCS object and the binning factor, and returns a new WCS object
    with the updated WCS information that corresponds to the rebinned image.

    Args:
        wcs (WCS): The original WCS object containing the WCS information from the FITS header before rebinning.
        factor (Tuple[int, int]): A tuple containing the binning factors for the Y and X dimensions, respectively. 
                                  For example, (2, 2) means that the image is rebinned by a factor of 2 in both dimensions.
    
    Returns:
        WCS: A new WCS object with the updated WCS information that corresponds to the rebinned image. 
        This includes adjustments to the CD matrix or CDELT values, as well as updates to the CRPIX values 
        if they are present in the original WCS.
    """
    fx, fy = factor
    new_wcs = wcs.deepcopy()
    #logger.debug(f"Original WCS CD matrix:\n{new_wcs.wcs.cd}\nOriginal WCS CDELT:\n{new_wcs.wcs.cdelt}\nOriginal WCS CRPIX:\n{new_wcs.wcs.crpix}")

    if new_wcs.wcs.has_cd(): # If the WCS has a CD matrix, we need to scale it by the binning factor.
        cd = new_wcs.wcs.cd
        cd[0, :] *= fy
        cd[1, :] *= fx
        new_wcs.wcs.cd = cd
    else:
        if new_wcs.wcs.cdelt is not None: # If the WCS has CDELT values, we need to scale them by the binning factor.
            new_wcs.wcs.cdelt[0] *= fy
            new_wcs.wcs.cdelt[1] *= fx
    
    # If the WCS has CRPIX values, we need to adjust them by the binning factor as well,
    # since the reference pixel will now correspond to a different location in the rebinned image.
    if new_wcs.wcs.crpix is not None: 
        new_wcs.wcs.crpix[0] = (new_wcs.wcs.crpix[0] - 0.5) / fx + 0.5
        new_wcs.wcs.crpix[1] = (new_wcs.wcs.crpix[1] - 0.5) / fy + 0.5

    return new_wcs

def reduce_binning(
    input_fits: Union[str, fits.HDUList, np.ndarray],
    factor: Union[int, Tuple[int, int]] = 2,
    func: Callable[[np.ndarray], np.ndarray] = np.mean,
    save_as: str | None = None,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Reduces the resolution of a FITS image by applying a binning (rebinning) process, 
    which combines blocks of pixels into single pixels according to a specified factor 
    and reduction function. The function also updates the FITS header to reflect 
    the new image dimensions and WCS information.

    Args:
        input_fits (Union[str, fits.HDUList, np.ndarray]): The input FITS image, which can be provided 
        as a file path (str), an HDUList object, or a 2D/3D NumPy array containing the image data.

        factor (Union[int, Tuple[int, int]], optional): The binning factor to be applied to the image. 
        This can be a single integer (e.g., 2) which applies the same factor to both dimensions (Y and X), or a 
        tuple of two integers (e.g., (2, 3)) which applies different factors to the Y and X dimensions respectively.
        Defaults to 2.

        func (Callable[[np.ndarray], np.ndarray], optional): The reduction function to be applied to the blocks 
        of pixels during the rebinning process. This function should take a NumPy array as input and return a single 
        value (e.g., np.mean, np.median, np.sum). Defaults to np.mean.
        
        save_as (str | None, optional): If provided, the rebinned image will be saved to the specified file path as 
                                        a new FITS file. If None, the rebinned image will not be saved to disk. 
                                        Defaults to None.

    Returns:
        Tuple[np.ndarray, fits.Header]: A tuple containing the rebinned image data as a NumPy array and the updated 
        FITS header as a fits.Header object. The header will include updated WCS information if the original header 
        contained WCS keywords, as well as new keywords indicating the binning factors used (e.g., 'XBIN' and 'YBIN').

    Raises:
        ValueError: If the input factor is not a positive integer or a tuple of two positive integers, or if the dimensions 
        of the input image are not divisible by the corresponding binning factors.
    """

    # Load the image data and header from the input FITS file or HDUList, or use the provided NumPy array directly.
    if isinstance(input_fits, str): # If the input is a file path, we need to open the FITS file and read the data and header.
        hdulist = fits.open(input_fits, memmap=False)
        data = hdulist[0].data
        header = hdulist[0].header
        hdulist.close()
    elif isinstance(input_fits, fits.HDUList): # If the input is an HDUList, we can directly access the data and header 
        data = input_fits[0].data              # from the first HDU.
        header = input_fits[0].header
    elif isinstance(input_fits, np.ndarray): # If the input is a NumPy array, we can use it directly as the image data, 
        data = input_fits                    # but we don't have a header in this case, so we create an empty one.
        header = fits.Header()               # The user can provide a header separately if needed.
    else:
        log_and_raise(ValueError("Input must be a file path, an HDUList, or a NumPy array."))

    if data is None:
        log_and_raise(ValueError("The input FITS file does not contain image data."))

    #Normalize the factor to ensure we have a tuple of two integers for the Y and X dimensions.
    fy, fx = _check_factor(factor)

    ny, nx = data.shape[-2:]           # accept 2D or 3D arrays, where the last two dimensions are Y and X.
    if ny % fy != 0 or nx % fx != 0:
        log_and_raise(ValueError(f"Image dimensions ({ny}, {nx}) are not divisible by the binning factors (Y: {fy}, X: {fx})."))

    # Apply block_reduce to perform the rebinning. The block_size is set according to the binning factors,
    # and the reduction function is applied to each block of pixels.
    rebinned = block_reduce(
        data,
        block_size=(fy, fx),
        func=func
    )

    # Update the FITS header to reflect the new image dimensions and WCS information, if applicable.
    new_header = header.copy()

    # Shape of the rebinned image is (ny//fy, nx//fx), so we need to update the NAXIS1 and NAXIS2 keywords accordingly.
    new_header["NAXIS1"] = rebinned.shape[-1]
    new_header["NAXIS2"] = rebinned.shape[-2]

    # If the original header contains WCS information, we need to update it to reflect the new pixel scale and reference pixel after rebinning.
    if "WCSAXES" in new_header or "CRVAL1" in new_header:
        w = WCS(header)
        w_new = _update_wcs(w, (fy, fx))
        # Overwrite the WCS keywords in the new header with the updated WCS information. 
        new_header.update(w_new.to_header())

    # Save the rebinned image to a new FITS file if a save path is provided, using the updated header.
    new_header['XBIN'] = fx
    new_header['YBIN'] = fy
    if save_as is not None:
        hdu = fits.PrimaryHDU(data=rebinned, header=new_header)
        hdu.writeto(save_as, overwrite=True)
        logger.info(f"Rebinned image saved to {save_as} with binning factors (Y: {fy}, X: {fx}).")

    return rebinned, new_header
