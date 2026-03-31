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

__author__="Fabricio M. Pérez-Toledo"
__version__ = "0.0.1"
__license__ = "GPL v3.0"

from check_files import *
from reduction_hcam import *
from THILOS.BKP_alignment_hcam import *

import argparse, time, os, shutil
import os, json, warnings
import pkg_resources
from pathlib import Path

from Color_Codes import bcolors as bcl
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


############## Predefined functions #############

def create_config_file_home():
    """
    This function creates a copy of the configuration file in .config/thilos
    / for easier accessibility.
    """
    config_path = pkg_resources.resource_filename(
    'THILOS', 'config/configuration.json')
    shutil.copy(config_path,Path(os.getcwd())/'configuration.json')
    print(f"{bcl.OKGREEN}Configuration file created successfully in the current directory.{bcl.ENDC}")
    sys.exit()

def readJSON() -> json:
    """
    Reads the file containing the configuration parameters.

    Returns:
        json: Collection of configuration parameters 
    """
    return json.load(open(Path(os.getcwd())/'configuration.json'))


def run(path_config: str = None):
    """
    This function 
    """
    # Parse configuration
    parser = argparse.ArgumentParser(
                         prog = 'THILOS',
                         description = 'This software reduces observations taken with HIPERCAM\
                            in Deep Field mode. It can process any filter configuration and is suitable\
                            for observations affected by fringing (Sloan_z).')

    parser.add_argument('-e', '--execute', help='Execute the configuration file in the current directory.',
                        action='store_true')
    
    parser.add_argument('-c', '--create_config', help='Create a configuration file in the current directory.',
                        action='store_true')
    
    parser.add_argument('-p', '--path_config', type=str, help='Path to the configuration file and raw data.',
                        default=None)

    args = parser.parse_args()


    print(f"{bcl.OKBLUE}***********************************************************************{bcl.ENDC}")
    print(f"{bcl.OKBLUE}************************* WELCOME TO THILOS **************************{bcl.ENDC}")
    print(f"{bcl.OKBLUE}***********************************************************************{bcl.ENDC}")
    print("\n")
    print(f"{bcl.BOLD}---------------------- LICENSE ----------------------{bcl.ENDC}")
    print("\n")
    print(f"This program is free software: you can redistribute it and/or modify\n\
it under the terms of the GNU General Public License as published by\n\
the Free Software Foundation, either version 3 of the License, or\n\
(at your option) any later version.\n\n\
This program is distributed in the hope that it will be useful,\n\
but WITHOUT ANY WARRANTY; without even the implied warranty of\n\
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\n\
GNU General Public License for more details.\n\n\
You should have received a copy of the GNU General Public License\n\
along with this program. If not, see <https://www.gnu.org/licenses/>.")
    print("\n")
    print(f"{bcl.BOLD}************************ IMPORTANT INFORMATION ************************{bcl.ENDC}")
    print("\n")
    print(f"This software is designed to reduce Deep Field Imaging observations obtained with HIPERCAM.\n\
For proper use, you need to modify the configuration file, which can be found\n\
in the directory where this software is installed. Additionally, you need to create\n\
an account on Astrometry.net. Once you have the code that allows you to use the API,\n\
you need to fill in the correct variable.")
    print(f"\n")

    # Check if the configuration file exists (2025-08-04)
    if args.create_config:
        logger.info(f"Creating configuration file...")
        logger.info(f"Configuration file created successfully in the current directory")
        create_config_file_home()
        sys.exit()

    conf = check_files(path_config=args.path_config)

    PRG = conf['PRG']
    OB = conf['OB']

    hora_local = time.localtime()
    print(conf)
    logger.add(Path(conf['DIRECTORIES']['PATH'])/f"thilos_{time.strftime('%Y-%m-%d_%H:%M:%S', hora_local)}.log", format="{time} {level} {message} ({module}:{line})", level="INFO",
            filter=lambda record: 'astropy' not in record["name"])
        
    logger.info(f'{bcl.OKGREEN}Log file created{bcl.ENDC}')
    logger.info(f'{bcl.OKGREEN}Configuration has been updated successfully{bcl.ENDC}')
    logger.info(f'{bcl.OKGREEN}Read the configuration file successfully{bcl.ENDC}')
    
    #Reduction Recipe. This recipe is responsible for cleaning the images by subtracting 
    #the masterbias and dividing by the normalized masterflat.
    #Subsequently, the cleaned images are saved.
    logger.info(f'{bcl.OKBLUE}---------- Starting the reductions for {PRG}-{OB} ----------{bcl.ENDC}')
    
    #?bpm_path = pkg_resources.resource_filename('THILOS', 'BPM/BPM_OSIRIS_PLUS.fits')
    bpm_path = "/home/fabricio.perez/Core/OBS-TECNICO/1_PROYECTOS_TECNICOS/T301_THILOS/THILOS/BPM/BPM_OSIRIS_PLUS.fits"
    o = Reduction(main_path=conf['DIRECTORIES']['PATH_DATA'],
                path_mask=bpm_path)
    o.get_imagetypes()
    o.load_BPM()
    o.check_binning()
    o.sort_down_drawer()

    if conf['REDUCTION']['use_BIAS']:
        o.do_masterbias()

    else:
        logger.warning(f'{bcl.WARNING}The masterbiases are not going to be created{bcl.ENDC}')

    if conf['REDUCTION']['use_FLAT']:
        o.do_masterflat()

    else:
        logger.warning(f'{bcl.WARNING}The masterflats are not going to be created{bcl.ENDC}')
    
    if conf['REDUCTION']['use_STD']:
        o.get_std(no_CRs=conf['REDUCTION']['no_CRs'], contrast_arg = conf['REDUCTION']['contrast'],
            cr_threshold_arg = conf['REDUCTION']['cr_threshold'],
            neighbor_threshold_arg = conf['REDUCTION']['neighbor_threshold'], apply_flat=conf['REDUCTION']['use_FLAT'])

    else:
        logger.warning(f'{bcl.WARNING}The STD star is not going to be reduced{bcl.ENDC}')
    
    o.get_target(no_CRs=conf['REDUCTION']['no_CRs'], contrast_arg = conf['REDUCTION']['contrast'],
            cr_threshold_arg = conf['REDUCTION']['cr_threshold'],
            neighbor_threshold_arg = conf['REDUCTION']['neighbor_threshold'], apply_flat=conf['REDUCTION']['use_FLAT'])
    
    
    if conf['REDUCTION']['save_fringing']:
        o.remove_fringing()
        logger.info(f'{bcl.OKGREEN}Fringing correction applied successfully{bcl.ENDC}')
    else:
        logger.warning(f'{bcl.WARNING}The fringing correction is not going to be executed{bcl.ENDC}')

    if conf['REDUCTION']['save_not_sky']:    
        o.sustract_sky()
        logger.info(f'{bcl.OKGREEN}The sky subtraction has been applied successfully{bcl.ENDC}')
    else:
        logger.warning(f'{bcl.WARNING}The sky substraction is not going to be executed{bcl.ENDC}')
    
    o.save_target(std=conf['REDUCTION']['save_std'])
    o.save_target(std=conf['REDUCTION']['save_std'], fringing=conf['REDUCTION']['save_fringing'])
    o.save_target(sky=conf['REDUCTION']['save_sky'])
    o.save_target(fringing=conf['REDUCTION']['save_fringing'])    
    o.save_target(not_sky=conf['REDUCTION']['save_not_sky'])
    logger.info(f'{bcl.OKBLUE}-------------- End of the reductions successfully --------------{bcl.ENDC}')
    print(2*"\n")

    #Aligned Recipe. The cleaned science images are aligned based on the filter used in each case. 
    #Then, they are saved as aligned images.
    if conf['ALIGNING']['use_aligning']:
        logger.info(f"{bcl.OKBLUE}---------- Starting the alignments ----------{bcl.ENDC}")
        aligner = AstroImageAligner(min_snr=10.0, conf = conf)
        for filt in list(set(aligner.ic.summary['filtro'])):
            for sky in ['SKY', 'NOSKY']:
                if conf['REDUCTION']['save_not_sky'] or sky == 'SKY':
                    logger.info(f'{bcl.OKCYAN}++++++++++ Aligment for {filt} & {sky} ++++++++++{bcl.ENDC}')
                    align = aligner.run_aligning(filt, sky=sky)
                    #lst = aligner._load_frames(filt, sky=sky)
                    #fr = CCDData.read(lst[0], unit='adu')
                    header = aligner.ref.header
                    header['STACKED'] = (True, 'Stacked image')
                    header['exptime'] = aligner.total_exptime * aligner.num #(al.num + 1.)
                    logger.info(f"Estimated total exposure time: {header['exptime']} sec")
                    wcs = aligner.fr.wcs
                    logger.info(f"Updating the WCS information")
                    save_fits(align, header, wcs, str(aligner.PATH_REDUCED / f'{PRG}_{OB}_{filt}_stacked_{sky}.fits'))
                    
                else:
                    logger.warning(f'{bcl.WARNING}Alignments are not going to be executed for NOSKY{bcl.ENDC}')

        logger.info(f'{bcl.OKBLUE}------------------- End of the alignments -------------------{bcl.ENDC}')
        print(2*"\n")
    else:
        logger.warning(f'{bcl.WARNING}The alignments are not going to be executed{bcl.ENDC}')

if __name__ == '__main__':
    run()
