import os
import warnings
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader


from .CodeLoaders import RolledCodes, UnrolledCodes
from .CacheHandler import CacheHandler

def find_device():
    device = 'cpu'
    if torch.cuda.is_available(): #check if NVIDIA gpu is availible
        device = 'cuda:0'
        warnings.warn("CUDA device found: defaulting to CUDA device 0.")
    elif torch.backends.mps.is_available():
        device = 'mps'
        warnings.warn("Apple Silicon found: defaulting to MPS.")
    else:
        warnings.warn("No CUDA device or Apple Silicon found. Operations may be slow...")

    return device


class Hamming:
    def __init__(self, hamming_params):
        self.__roll =  hamming_params['roll'] #bool
        self.__self_comparison = hamming_params['self_comparison'] #bool
        self.__linear = hamming_params['linear'] #bool
        self.__verbose = hamming_params['verbose'] #bool
        self.__reference_batch_size = hamming_params['reference_batch_size'] #int
        self.__comparison_batch_size = hamming_params['comparison_batch_size'] #int
        self.__data_path = hamming_params['data_path'] #str
        self.__results_path = hamming_params['results_path'] #str
        self.__device = find_device() #str

        self.__reference_loader = self.__set_loader(0, True) #True to indicate that reference is being set

    
    def calculate(self):  #TODO nest tqdm
        #warnings.warn("Calculating hamming distances may take hours or days...") warn about amount of drive space that will be used for caching
        warnings.warn("Calculating hamming distances may take hours or days...")
        for i, output in tqdm(enumerate(self.__data_path)):
            comparison_loader = None
            Cache = None
            target = f'{self.__results_path}/reference-{output.rsplit("/", 2)[1]}__{output.rsplit("/", 2)[2].replace("-GABOR_EXTRACTED.npz", "")}'
            os.makedirs(target, exist_ok=True)
            if output == self.__data_path[0] and (self.__verbose or self.__self_comparison):
                comparison_loader = self.__set_loader(i, False)
                Cache = CacheHandler(target, True)
            elif output == self.__data_path[0]:
                continue
            else:
                comparison_loader = self.__set_loader(i)
                Cache = CacheHandler(target, False)
            
            self.__pairwise_hamming_distance(comparison_loader, Cache)

            if self.__verbose:
                Cache.save()
                Cache.clear()



        #sets loaders
    def __set_loader(self, i, reference=False):
        dataset = None
        loader_batch_size = None
        codes = np.load(self.__data_path[i], allow_pickle=True)
        if reference:
            loader_batch_size = self.__reference_batch_size
        else:
            loader_batch_size = self.__comparison_batch_size

        if reference and self.__roll:
            dataset = RolledCodes(codes)
        else:
            dataset = UnrolledCodes(codes)
        
        loader = DataLoader(dataset, batch_size=loader_batch_size, pin_memory=True, shuffle=False)

        return loader
    
    def __pairwise_hamming_distance(self, comparison_loader, Cache):
        for reference_batch in tqdm(self.__reference_loader):
            size = reference_batch[0].shape[-1]
            reference_batch_gpu = reference_batch[0].to(self.__device)
            reference_batch_gpu = reference_batch_gpu.unsqueeze(1)
            if self.__roll:
                for comparison_batch in comparison_loader:
                    comparison_batch_gpu = comparison_batch[0].to(self.__device)
                    comparison_batch_gpu = comparison_batch_gpu.unsqueeze(1)
                    result = torch.bitwise_xor(reference_batch_gpu, comparison_batch_gpu).sum(dim=-1).min(dim=-1)[0]
                    for i in range(len(reference_batch[1])):
                        for j in range(len(comparison_batch[1])):
                            proto_string = [reference_batch[1][i], comparison_batch[1][j]]
                            proto_string.sort()
                            Cache.new_line(f'{proto_string[0]}|{proto_string[1]}', result[i][j]/size)
            else:
                for comparison_batch in comparison_loader:
                    comparison_batch_gpu = comparison_batch[0].to(self.__device)
                    comparison_batch_gpu = comparison_batch_gpu.unsqueeze(0)
                    result = torch.bitwise_xor(reference_batch_gpu, comparison_batch_gpu).sum(dim=-1)
                    for i in range(len(reference_batch[1])):
                        for j in range(len(comparison_batch[1])):
                            proto_string = [reference_batch[1][i], comparison_batch[1][j]]
                            proto_string.sort()
                            Cache.new_line(f'{proto_string[0]}|{proto_string[1]}', result[i][j]/size)