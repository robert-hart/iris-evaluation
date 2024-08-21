"""
Code for "generated_iris_evaluation".
"""

from .core.AnalysisArgs import AnalysisArgs
from .core.MasekCode import MasekGaborKernel, MasekGaborResponse
from .core.CodeLoaders import RolledCodes, UnrolledCodes
from .core.Normalize import CoordinateTransformer

from .core.hamming import pairwise_hamming_distance, set_torch_device
from .core.feature_extraction import process_dataset, make_instructions

from .utils.elementary_segmentation import elementary_segmentation, otsu_thresh