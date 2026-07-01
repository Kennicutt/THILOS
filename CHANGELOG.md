# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
