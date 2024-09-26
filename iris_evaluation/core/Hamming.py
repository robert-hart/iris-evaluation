#TODO: better OOP & algs.
import os
import warnings
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader


from .CodeLoaders import RolledCodes, UnrolledCodes
from .CacheHandler import PairwiseCache, LinearCache

def find_device():
    device = 'cpu'
    if torch.cuda.is_available(): #check if NVIDIA gpu is availible
        device = 'cuda:0'
        print("CUDA device found: defaulting to CUDA device 0.\n")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Apple Silicon found: defaulting to MPS.\n")
    else:
        warnings.warn("No CUDA device or Apple Silicon found. Operations may be slow...\n")

    return device


class Hamming(object):
    def __init__(self, hamming_params):
        self.__comparison_batch_size = None

        self.__roll =  bool(hamming_params['roll']) #bool whether or not to roll
        self.__verbose = bool(hamming_params['verbose']) #bool
        self.__pairwise = bool(hamming_params['pairwise'])
        self.__reference_batch_size = int(hamming_params['reference_batch_size']) #int
        self.__data_paths = hamming_params['data_paths'] #str
        self.__reference_tag = self.__data_paths[0].rsplit("__", 1)[0].split("/")[-1]
        self.__results_path = hamming_params['results_path'] #str
        self.__device = find_device() #str
        self.__reference_loader = self.__set_loader(0, True)

    def calculator(self):  #TODO nest tqdm
        num_comparisons = None
        num_sets = len(self.__data_paths)
        if self.__verbose:
            num_comparisons = (num_sets * 2) - 1
        else:
            num_comparisons = num_sets
        count = 1

        print("Comparing datasets. May take take some time...\n")

        for i, data_path in enumerate(self.__data_paths):
            if self.__pairwise:
                self.__comparison_batch_size = int(self.__reference_batch_size * 3)
                if data_path == self.__data_paths[0]:
                    print(f"\n(COMPARISON {count}/{num_comparisons}):\tCalculating intra-dataset pairwise hamming distances for {self.__reference_tag}.\n")
                    self.__calculate_self_pairwise(i, self.__reference_tag)
                    count = count + 1
                else:
                    dataset_tag = data_path.rsplit("__", 1)[0].split("/")[-1]
                    print(f"\n(COMPARISON {count}/{num_comparisons}):\tCalculating inter-dataset pairwise hamming distances between {self.__reference_tag} & {dataset_tag}.\n")
                    self.__calculate_pairwise(i, dataset_tag)
                    count = count + 1
                    if self.__verbose:
                        print(f"\n(COMPARISON {count}/{num_comparisons}):\tCalculating intra-dataset pairwise hamming distances for {dataset_tag}.\n")
                        self.__calculate_self_pairwise(i, dataset_tag)
                        count = count + 1
            else:
                self.__calculate_linear(i, data_path) #TODO: double check this whole branch
    """
    METHODS TO HELP WITH LOGISTICS, PROBABLY OVERCOMPLICATED BUT IT WORKS.
    """

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
    
    def __calculate_pairwise(self, i, dataset_tag):
        comparison_loader = self.__set_loader(i)
        target = f'{self.__results_path}/pairwise-inter__{self.__reference_tag}_&_{dataset_tag}'
        os.makedirs(target, exist_ok=True)
        Cache = PairwiseCache(target, True, False)
        self.__pairwise_hamming_distance(self.__reference_loader, comparison_loader, Cache)
        Cache.save()
        Cache.clear()

    #save in data_path
    def __calculate_self_pairwise(self, i, dataset_tag):
        reference_loader = None
        if i == 0:
            reference_loader = self.__reference_loader
        else:
            reference_loader = self.__set_loader(i, reference=True)

        comparison_loader = self.__set_loader(i)

        target = f'{self.__results_path}/pairwise-intra__{dataset_tag}_&_{dataset_tag}'
        os.makedirs(target, exist_ok=True)

        Cache = PairwiseCache(target, True, True)
        self.__pairwise_hamming_distance(reference_loader, comparison_loader, Cache)
        Cache.save()
        Cache.clear()
    
    def __calculate_linear(self, i, data_path):
        set_tag = data_path.split("__", 1)[0]
        comparison_loader = self.__set_loader(i)

        target = f'{self.__results_path}/linear__{self.__reference_tag}_&_{set_tag}'
        os.makedirs(target, exist_ok=True)

        Cache = LinearCache(target, False)
        self.__linear_hamming_distance(comparison_loader, Cache)
        Cache.save()
        Cache.clear()

    def __find_conditions(self, comparison_names):
        conditions = []
        for comparison_name in comparison_names:
            condition = None
            try:
                condition = comparison_name.split('___cond___')[1]
            except:
                condition = 'NA_NA'
            
            conditions.append(condition)
    
        return tuple(conditions)
    
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
    
    def __linear_hamming_distance(self, comparison_loader, Cache):
        for (reference_codes, reference_names), (comparison_codes, comparison_names) in tqdm(zip(self.__reference_loader, comparison_loader)):
            reference_codes = reference_codes.to(self.__device)
            reference_names = reference_names.to(self.__device)
            comparison_codes = comparison_codes.to(self.__device)
            comparison_names = comparison_names.to(self.__device)

            comparison_codes_local = None
            hamming_distances = None

            if self.__roll:
                comparison_codes_local = comparison_codes.unsqueeze(1).expand(-1, reference_codes.shape[1], -1)
                xor_result = torch.bitwise_xor(reference_codes, comparison_codes_local)
                HD = xor_result.sum(dim=2)
                hamming_distances, _ = torch.min(HD, dim=1)
            else:
                xor_result = torch.bitwise_xor(reference_codes, comparison_codes_local)
                hamming_distances = xor_result.sum(dim=2)

            conditions = self.__find_conditions(comparison_names)

        for i in range(reference_codes.shape[0]):
            proto_string = f'{reference_names[i]}|{conditions[i]}'
            Cache.new_line(f'{proto_string}', hamming_distances[i])