# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-09-25

- Added a new functions to remove fringing from the images with dithering and without dithering.
- Allow to omit the reduction step.
- Added a new function to filter saturated frames before combining them to create the master flat.
- Added new fringing maps for binning 4x4.

## [0.2.0] - 2026-07-16

- Fixed a bug that the thilos didn't save frames with illumination correction applied.
- Improved log message for masterbias creation. 

## [0.1.6] - 2026-07-06

- The BPMs cannot be distributed due to a decompression issue in PyPI. They must be downloaded from the GTC website.

## [0.1.5] - 2026-07-06

- Everything OK

## [0.1.4] - 2026-07-06

- Error with PYPI.

## [0.1.3] - 2026-07-06

 - Fixed a bug in the BPM path.

## [0.1.2] - 2026-07-06

- Added illumination correction.
- Added BPM correction for binning 1x1 and 2x2. This include a new algorithm to fill bad pixels by the value of adyacent pixels.
- Fixed a bug in removing fringing.

## [0.1.1] - 2026-07-01

### Added
- Added a script to rename the joinup output file. The new format allows to THILOS to read the file and continue with the reduction process.

Usage:
    $ cd path/to/frames
    $ python -m THILOS.script.rename_joinup_fits

## [0.1.0] - 2026-07-01

### Added
- Initial release on PyPI.
- Console script entry point: `thilos` command.
- Support for reducing HIPERCAM Deep Field Imaging observations. 
