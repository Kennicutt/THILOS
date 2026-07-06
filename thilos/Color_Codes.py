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

class bcolors:
    """
    A class to define string color formatting for terminal output using ANSI escape sequences.

    This class provides various color codes and text styles that can be used to format terminal output. 
    The colors and styles are defined as class variables, and each variable represents a different format 
    that can be applied to text in the terminal.

    Class variables:
        HEADER (str): Color for header text.
        OKBLUE (str): Color for informational text (blue).
        OKCYAN (str): Color for informational text (cyan).
        OKGREEN (str): Color for successful operation text (green).
        WARNING (str): Color for warning text (yellow).
        FAIL (str): Color for error or failed operation text (red).
        ENDC (str): Resets the text formatting to default.
        BOLD (str): Bold text style.
        UNDERLINE (str): Underlined text style.
    """
    HEADER = '\033[33m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'