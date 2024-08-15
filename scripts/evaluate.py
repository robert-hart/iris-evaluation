import os
import argparse
from tqdm import tqdm
import multiprocessing

import numpy as np
from torch.utils.data import DataLoader

from iris_evaluation import MasekGaborKernel, AnalysisArgs, make_instructions, ReferenceSet, SampleSet, hamming_distance, set_device, process_dataset


#options 0, 1, 2
#0 = no segmentation

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
                Gabor = MasekGaborKernel(process_parameters[3], int(args.wavelength), int(args.octaves), process_parameters[4])
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
    
    if args.comparison_mode:
        #
        reference_batch_size = 4
        sample_batch_size = 10

        device = set_device()

        target_paths = []
        target_dir = f'{args.comparison[0].rsplit("/", 2)[0]}/results'
        os.makedirs(target_dir, exist_ok=True)

        #load reference set
        reference_codes = np.load(args.comparison[0], allow_pickle=True)
        reference_dataset = ReferenceSet(reference_codes)
        reference_data = DataLoader(reference_dataset, batch_size=reference_batch_size, pin_memory=True)

        #calculate intra group hamming distance
        reference_unrolled = DataLoader(SampleSet(reference_codes), batch_size=reference_batch_size, pin_memory=True)




        #load comparison sets
        for output in tqdm(args.comparison):
            if output == args.comparison[0]:
                continue
            else:
                target = f'{target_dir}/reference-{output.rsplit("/", 2)[1]}__{output.rsplit("/", 2)[2].replace("-GABOR_EXTRACTED.npz", "")}'
                #TODO delete anything in target folder if it exists
                os.makedirs(target, exist_ok=True)
                target_paths.append(target)

                comparison_codes = np.load(output, allow_pickle=True)
                comparison_dataset = SampleSet(comparison_codes)
                comparison_data = DataLoader(comparison_dataset, batch_size=sample_batch_size, pin_memory=True)

                hamming_distance(device, f'{target}/hamming.txt', reference_data, comparison_data)

                #calculate intra group hamming distances / unrolled

                #organize

                #cutoffs




        
if __name__ == '__main__':
    manager = multiprocessing.Manager()
    main()