import os
import argparse
from tqdm import tqdm
import multiprocessing
import numpy as np
import MasekCode as mk

from gabor_extract import process_dataset, make_instructions
from AnalysisArgs import AnalysisArgs

#TODO make calibrate an option
#TODO consolidate helper functions

def main():
    #reads the csv
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

                process_parameters = (1, 256, 45, 360, 2, bool(args.verbose)) #parameters of the dataset | (whether to segment, input size (square), output y, output x), multiplicative factor, verbose

                instructions = make_instructions(source_images, target_path)
                Gabor = mk.MasekGaborKernel(process_parameters[3], int(args.wavelength), int(args.octaves), process_parameters[4])
                gabor_kernels = Gabor.kernels #kernels

                #TODO save parameters in dir
                gabor_parameters = Gabor.parameters #kernel information parameters | ksize, signmaOnf/wavelength, wavelength

                pool = multiprocessing.Pool(processes=int(args.threads))
                data_shared = manager.list([process_parameters, gabor_kernels])

                with tqdm(total=len(instructions)) as pbar:
                    for result in pool.imap_unordered(process_dataset, [(data_shared, instruction) for instruction in instructions]):                
                            pbar.update()
                    pool.close()
                    pool.join()

        #create NPZ file & set the standard for the directory
        #TODO if not verbose, delete the individual iris codes
        for target_path in target_paths:
            print(f'consolidating {target_path}...')
            all_codes = {}
            for npy in os.listdir(target_path):
                if '.npy' in npy:
                    name = npy.split('.')[0]
                    all_codes[name] = np.load(f'{target_path}/{npy}', allow_pickle=True)
            np.savez(f'{target_path}-GABOR_EXTRACTED.npz', **all_codes)
            print(f'consolidation complete')
    
    #change to hamming distance
    #find all paths
    if args.comparison_mode:

    #if intra-dataset comparison is desired

        #TODO comparison mode
        pass

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    main()