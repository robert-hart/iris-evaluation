"""
Code for "iris_evaluation".
"""

from .core.MasekCode import MasekGaborKernel, MasekGaborResponse
from .core.CodeLoaders import RolledCodes, UnrolledCodes
from .core.Normalize import CoordinateTransformer
from .core.CacheHandler import PairwiseCache, LinearCache

from .core.Hamming import Hamming
from .core.FeatureExtraction import FeatureExtraction

from .utils.elementary_segmentation import elementary_segmentation, otsu_thresh