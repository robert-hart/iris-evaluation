"""
Code for "generated_iris_evaluation".
"""

from .core.AnalysisArgs import AnalysisArgs
from .core.MasekCode import MasekGaborKernel, MasekGaborResponse
from .core.DataLoaders import ReferenceSet, SampleSet
from .core.Normalize import CoordinateTransformer

from .utils.segmentation import segmentation, otsu_thresh

from .routines.gabor import process_dataset, make_instructions
from .routines.hamming import hamming_distance, set_device