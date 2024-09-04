import os
import argparse
from tqdm import tqdm
import multiprocessing
import numpy as np

from iris_evaluation import AnalysisArgs, FeatureExtraction, Hamming


def main():
    #reads the csv, make read JSON instead
    arg_parser = argparse.ArgumentParser(description='Finds iris codes for all iris images in a dataset')
    arg_parser.add_argument('-p', '--path', help='path to flag csv', required=False)
    args_path = arg_parser.parse_args()

    args = AnalysisArgs(args_path.path)

    #if gabor extraction is desired
    if args.gabor_extraction: #run gabor extraction
        target_paths = []
        for source in args.all_sources:
            if source != None:
                source_images = np.load(f'{source}', allow_pickle=True) #load the source images
                target_path = f'{source}'.replace('.npz', '') #get the target path
                target_paths.append(target_path)
                os.makedirs(target_path, exist_ok=True)

                logistical_parameters = {
                    'source': source_images, #maybe just source?
                    'target_path': target_path,
                    'threads': args.threads
                }

                shared_parameters = {
                    "segment": 1,
                    "multiplicative_factor": 2,
                    "image_size": 256,
                    "output_y": 45,
                    "output_x": 360,
                    "verbose": True,
                    "wavelength": int(args.wavelength),
                    "octaves": int(args.octaves),
                }

                FE = FeatureExtraction(logistical_parameters, shared_parameters)
                FE.calculate()
                FE.clean()

    if args.comparison_mode:
        results_path = f'{args.comparison[0].rsplit("/", 2)[0]}/results'
        os.makedirs(results_path, exist_ok=True)

        hamming_params = {
            "roll" : True,
            "verbose" : True,
            "pairwise" : True,
            "data_paths" : args.comparison,
            "results_path" : results_path,
            "reference_batch_size" : 4,
        }

        HammingSetup = Hamming(hamming_params)
        HammingSetup.calculator()

if __name__ == '__main__':
    main()