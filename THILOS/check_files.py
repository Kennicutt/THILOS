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

import json, os
#!import pkg_resources
from importlib.resources import as_file, files
import ccdproc as ccdp
from pathlib import Path
from typing import Dict

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

def read_config(config_path: str) -> Dict:
    """
    Reads a configuration file from the given path and returns it as a dictionary.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict: The loaded configuration as a dictionary.
    """
    logger.info(f'Reading configuration from {config_path}')
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def readJSON() -> json:
    """
    Reads the file containing the configuration parameters.

    Returns:
        json: Collection of configuration parameters 
    """
    if os.path.exists(Path(os.getcwd())/'configuration.json'):
        logger.info('Configuration file found in the current working directory.')
        return json.load(open(Path(os.getcwd())/'configuration.json'))
    else:
        #!config_path = pkg_resources.resource_filename(
        #!    'THILOS', 'config/configuration.json')
        with as_file(files('THILOS').joinpath('config/configuration.json')) as config_path:
            # config_path es un Path object temporal
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f'Configuration file not found in the current working directory. Reading from {config_path}')
                return config
        #!logger.info(f'Configuration file not found in the current working directory. Reading from {config_path}')
        #!return json.load(open(config_path))


def update_config(config_path: str, config: Dict) -> Dict:
    """
    Updates the configuration file at the specified path with the provided configuration dictionary.

    Args:
        config_path (str): Path to the configuration file to be updated.
        config (Dict): The updated configuration dictionary.

    Returns:
        Dict: The updated configuration dictionary.
    """
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)
    logger.info('Configuration file updated.')
    
    return config


def classify_images(tab: ccdp.CCDData) -> Dict:
    """
    Classifies images based on their 'OBJECT' keyword in the provided CCDData table.

    Args:
        tab (ccdp.CCDData): The CCDData table containing image metadata.

    Returns:
        Dict: A dictionary indicating the existence of different image types.
    """
    existence = {
        'exist_BIAS': False,
        'exist_SKYFLAT': False,
        'exist_SCIENCE': False,
        'exist_STD': False
    }

    if 'Bias' in tab['OBJECT']:
        existence['exist_BIAS'] = True
        logger.info('Bias frames found in the directory.')
    
    if 'Skyflat' in tab['OBJECT']:
        existence['exist_SKYFLAT'] = True
        logger.info('Skyflat frames found in the directory.')

    sub = tab[(tab['OBJECT']!='Bias') & (tab['OBJECT']!='Skyflat')]
    existence['exist_STD'] = (len([item for item in sub['OBJECT'] if item.startswith('STD')]) != 0)
    logger.info(f"Standard star frames found in the directory: {existence['exist_STD']}")
    existence['exist_SCIENCE'] = (len([item for item in sub['OBJECT'] if not item.startswith('STD')]) != 0)
    logger.info(f"Science frames found in the directory: {existence['exist_SCIENCE']}")

    return existence


def check_files(path_config: str = None) -> json:
    """
    Checks the existence of different image types in the specified configuration path.

    Args:
        path_config (str, optional): The path to the configuration directory. Defaults to None.

    Returns:
        json: The updated configuration dictionary.
    """
    conf = readJSON()
    #!conf = read_config(path_config + 'configuration.json') #---> Temporal HAY QUE USAR EL readJSON() ORIGINAL
    directory = Path(os.getcwd())/'raw'
    #!directory = path_config +'raw'  #---> Temporal HAY QUE USAR EL os.getcwd() ORIGINAL

    ic = ccdp.ImageFileCollection(directory, keywords=['GTCPRGID','GTCOBID','OBSMODE','OBJECT','FILTER2','EXPTIME'])
    image_types = classify_images(ic.summary)

    conf['DIRECTORIES']['PATH'] = str(Path(os.getcwd())) #! path_config
    conf['DIRECTORIES']['PATH_DATA'] = str(Path(os.getcwd())/'raw') #! path_config + 'raw'
    conf['DIRECTORIES']['PATH_OUTPUT'] = str(Path(os.getcwd())/'reduced') #! path_config + 'reduced' 

    # Update config based on image types
    if conf['REDUCTION']['use_BIAS']:
        logger.info('The masterbias will be created and used in the reductions.')
        conf['REDUCTION']['use_BIAS'] = image_types['exist_BIAS']
    else:
        logger.info('The masterbiases are not going to be created.')
    
    if conf['REDUCTION']['use_FLAT']:
        logger.info('The masterflat will be created and used in the reductions.')
        conf['REDUCTION']['use_FLAT'] = image_types['exist_SKYFLAT']
    else:
        logger.info('The masterflats are not going to be created.')

    if conf['REDUCTION']['use_STD']:
        logger.info('The standard star frames will be used in the reductions.')
        conf['REDUCTION']['use_STD'] = (image_types['exist_STD'] and image_types['exist_SKYFLAT'] and conf['REDUCTION']['use_FLAT'])
    else:
        logger.info('The standard star frames will not be used in the reductions.')

    if conf['REDUCTION']['save_std']:
        logger.info('The reduced standard star frames will be saved.')
        conf['REDUCTION']['save_std'] = (image_types['exist_STD'] and image_types['exist_SKYFLAT'] and conf['REDUCTION']['use_STD'])
    else:
        logger.info('The reduced standard star frames will not be saved.')
    
    if conf['REDUCTION']['save_sky']:
        logger.info('The reduced science frames with sky subtraction will be saved.')
        conf['REDUCTION']['save_sky'] = image_types['exist_SCIENCE']
    else:
        logger.info('The reduced science frames with sky subtraction will not be saved.')

    if conf['REDUCTION']['save_fringing']:
        logger.info('The reduced science frames with fringing correction will be saved.')
    else:
        logger.info('The reduced science frames with fringing correction will not be saved.')

    return update_config(conf['DIRECTORIES']['PATH'] + '/configuration.json', conf)