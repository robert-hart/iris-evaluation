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
        print("CUDA device found: defaulting to CUDA device 0.")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Apple Silicon found: defaulting to MPS.")
    else:
        warnings.warn("No CUDA device or Apple Silicon found. Operations may be slow...")

    return device


class PairwiseHamming(object):
    def __init__(self, hamming_params):
        self.__roll =  bool(hamming_params['roll']) #bool whether or not to roll
        self.__verbose = bool(hamming_params['verbose']) #bool
        self.__reference_batch_size = int(hamming_params['reference_batch_size']) #int
        self.__comparison_batch_size = int(self.__reference_batch_size * 3)
        self.__data_paths = hamming_params['data_paths'] #str
        self.__results_path = hamming_params['results_path'] #str
        self.__device = find_device() #str
        self.__reference_loader = self.__set_loader(0, True)

    def calculate(self):  #TODO nest tqdm
        print("Calculating hamming distances. May take hours or days...")

        for i, data_path in tqdm(enumerate(self.__data_paths)):
            target = None
            comparison_loader = None
            Cache = None

            if data_path == self.__data_paths[0]: #no self comparison if not needed
                if not self.__verbose:
                    continue
                else:
                   self.__self_comparison(i, data_path)
                   continue

            comparison_loader = self.__set_loader(i)

            target = f'{self.__results_path}/reference-{data_path.rsplit("/", 2)[1]}__{data_path.rsplit("/", 2)[2].replace("-GABOR_EXTRACTED.npz", "")}'
            os.makedirs(target, exist_ok=True)

            Cache = CacheHandler(target, False) #not a self comparison
            self.__pairwise_hamming_distance(comparison_loader, Cache)
            Cache.save()
            Cache.clear()

            if self.__verbose:
                self.__self_comparison(i, data_path)


    def __self_comparison(self, i, data_path):
        reference_loader = None

        if i == 0:
            reference_loader = self.__reference_loader
        else:
            reference_loader = self.__set_loader(i, reference=True)

        comparison_loader = self.__set_loader(i)

        target = f"{data_path.replace('-GABOR_EXTRACTED.npz', f'_{i}_')}/self_comparison"
        os.makedirs(target, exist_ok=True)

        Cache = CacheHandler(target, True)
        self.__pairwise_hamming_distance(reference_loader, comparison_loader, Cache)
        Cache.save()
        Cache.clear()


    def __set_loader(self, i, reference=False):
        dataset = None
        batch_size = None
        codes = np.load(self.__data_paths[i], allow_pickle=True)
        if reference:
            batch_size = self.__reference_batch_size
        else:
            batch_size = self.__comparison_batch_size

        if reference and self.__roll:
            dataset = RolledCodes(codes)      
        else:
            dataset = UnrolledCodes(codes)
        
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

        return loader

    """
    OVERLOADED METHOD FOR PAIRWISE HAMMING DISTANCE TO CHANGE REFERENCE SOURCE
    TODO: make if/then statements more efficient. too much boilerplate.
    """

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
    

    def __pairwise_hamming_distance(self, reference_loader, comparison_loader, Cache):
        for reference_batch in tqdm(reference_loader):
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
    """
    LINEAR HAMMING DISTANCE

    TODO: test in notebook
    """
    #call, compare, repeat
    class LinearHamming(object):
        def __init__(self, hamming_params):
            pass
        
    def __linear_hamming_distance(self, comparison_loader, Cache): #reverse reference & comparison
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