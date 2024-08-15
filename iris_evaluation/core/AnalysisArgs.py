import os
import pandas as pd

#TODO return as correct type

class AnalysisArgs:
    def __init__(self, args_path):
        # input & output
        self.reference = None  # path to reference set
        self.comparison = None  # path to comparison set(s) for comparison operation
        self.all_sources = None # reference + comparison
        self.target = None  # path to output
        self.verbose = False  # intermediate outputs

        # functionality
        self.gabor_extraction = False
        self.comparison_mode = False

        # comparison parameters
        self.pairwise = False
        self.lpips = False
        self.ssim = False

        # gabor filter parameters
        self.wavelength = None
        self.octaves = None

        # computational parameters
        self.threads = None
        self.batch_size = None

        self._set_values(args_path)
    
    def _set_values(self, args_path):
        args_values = pd.read_csv(args_path, usecols=['values'])['values'].to_list()
        self.reference = args_values[0]
        
        self.target = args_values[1].rsplit('/', 1)[-1]  # parent directory of output
        self.verbose = bool(int(args_values[2]))

        self.gabor_extraction = bool(int(args_values[3]))
        self.comparison_mode = bool(int(args_values[4]))
        self.pairwise = bool(int(args_values[5]))
        self.lpips = bool(int(args_values[6]))
        self.ssim = bool(int(args_values[7]))

        self.wavelength = args_values[8]
        self.octaves = args_values[9]

        self.threads = args_values[10]
        self.batch_size = (int(args_values[11]), int(args_values[12]))

        comparison = []
        all_sources = []

        for npz in os.listdir(self.reference.rsplit('/', 1)[0]):
            if ".npz" in npz and "-GABOR_EXTRACTED.npz" in npz:
                comparison.append(f'{self.reference.rsplit("/", 1)[0]}/{npz}')
        
        for npz in os.listdir(args_values[1]):
            if ".npz" in npz and "-GABOR_EXTRACTED.npz" not in npz:
                all_sources.append(f'{args_values[1]}/{npz}')
            if "GABOR_EXTRACTED.npz" in npz:
                comparison.append(f'{args_values[1]}/{npz}')
        if len(comparison) == 0:
            comparison.append('NA')
            all_sources.append('NA')

        all_sources.append(args_values[0])

        self.comparison = tuple(comparison)
        self.all_sources = tuple(all_sources)