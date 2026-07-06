#!/usr/bin/env python3
"""
FITS filename renaming script for HIPERCAM data.
Renames files based on filename components and FITS header OBJECT field.
"""

import os
import re
import glob
from astropy.io import fits


def parse_original_filename(filename):
    """
    Parse the original filename to extract components.
    Format: ID-DATE-INSTRUMENT-TYPE.fits_ccdN_XXXX.fits
    """
    # Remove .fits extension for parsing
    base = filename.replace('.fits', '')
    
    prefix, ccd_num, frame_num = base.split('_')
    
    # Extract the observation type from the prefix
    # Format: ID-DATE-INSTRUMENT-TYPE
    type_match = re.search(r'-(Bias|SkyFlat|Imaging)$', prefix)
    if type_match:
        obs_type = type_match.group(1)
    else:
        raise ValueError(f"Could not determine observation type from {filename}")
    
    return {
        'obs_type': obs_type,
        'ccd_num': ccd_num,
        'frame_num': frame_num,
        'original': filename
    }


def determine_new_type(obs_type, object_value):
    """
    Determine the new observation type based on original type and OBJECT header.
    
    STD frames are originally labeled as 'Imaging' but contain 'STD_' in OBJECT.
    """
    if obs_type == 'Imaging':
        if object_value and object_value.startswith('STD_'):
            return 'STD'
        else:
            return 'Imaging'
    else:
        # Bias and SkyFlat remain unchanged
        return obs_type


def construct_new_filename(new_type, ccd_num, frame_num):
    """
    Construct the new filename in the format: TYPE_ccdN_XXXX.fits
    """
    return f"{new_type}_{ccd_num}_{frame_num}.fits"


def process_file(filepath):
    """
    Process a single FITS file: read header, determine new name, rename.
    """
    filename = os.path.basename(filepath)
    
    try:
        # Parse original filename
        parsed = parse_original_filename(filename)
        
        # Read FITS header to get OBJECT field
        with fits.open(filepath) as hdul:
            header = hdul[0].header #type: ignore
            object_value = header.get('OBJECT', '').strip()
        
        # Determine new type
        new_type = determine_new_type(parsed['obs_type'], object_value)
        
        # Construct new filename
        new_filename = construct_new_filename(
            new_type, 
            parsed['ccd_num'], 
            parsed['frame_num']
        )
        
        # Generate full paths
        directory = os.path.dirname(filepath)
        new_filepath = os.path.join(directory, new_filename)
        
        # Perform rename
        os.rename(filepath, new_filepath)
        
        print(f"Renamed: {filename}")
        print(f"  -> {new_filename}")
        if parsed['obs_type'] == 'Imaging':
            print(f"  OBJECT='{object_value}' -> type={new_type}")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False


def main():
    """
    Main function to process all FITS files in current directory.
    """
    # Find all matching FITS files
    pattern = "*HIPERCAM-*.fits_ccd*_*.fits"
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No matching FITS files found in current directory.")
        print("Expected pattern: *HIPERCAM-*.fits_ccd*_*.fits")
        return
    
    print(f"Found {len(files)} file(s) to process.\n")
    
    success_count = 0
    for filepath in files:
        if process_file(filepath):
            success_count += 1
    
    print(f"Done. Successfully renamed {success_count}/{len(files)} files.")


if __name__ == "__main__":
    main()