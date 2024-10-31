# Copyright 2024 USRA, NASA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
# import all types from typing

#this is set to 1 by default.
#it causes numba to spit out underutilization warnings
__global_grid_size__ = 1
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)