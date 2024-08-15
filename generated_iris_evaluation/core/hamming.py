from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from CodeLoader import ReferenceSet, SampleSet

def append_text_to_file(file_path, comparison, result):
    with open(file_path, 'a') as file:
        file.write(f"{comparison}\t{result}\n")

reference_batch_size = 4
sample_batch_size = 10

codes = np.load('iris_codes.npy', allow_pickle=True)
meta = np.load('metadata.npy', allow_pickle=True)

reference_dataset = ReferenceSet(codes, meta)
sample_dataset = SampleSet(codes, meta)

reference_data = DataLoader(reference_dataset, batch_size=reference_batch_size, pin_memory=True) 
sample_data = DataLoader(sample_dataset, batch_size=sample_batch_size, pin_memory=True)

def set_device():
    device = 'CPU'
    if torch.cuda.is_available(): #check if NVIDIA gpu is availible
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return device


def hamming_distance(device, cache_path, reference_data, sample_data):
    for reference_batch in tqdm(reference_data):
        size = reference_batch[0].shape[-1]
        reference_batch_gpu = reference_batch[0].to(device)
        reference_batch_gpu = reference_batch_gpu.unsqueeze(1)
        for sample_batch in sample_data:
            sample_batch_gpu = sample_batch[0].to(device)
            sample_batch_gpu = sample_batch_gpu.unsqueeze(1)
            result = torch.bitwise_xor(reference_batch_gpu, sample_batch_gpu).sum(dim=-1).min(dim=-1)[0]
            for i in range(len(reference_batch[1])):
                for j in range(len(sample_batch[1])):
                    proto_string = [reference_batch[1][i], sample_batch[1][j]]
                    proto_string.sort()
                    append_text_to_file(cache_path, f'{proto_string[0]}|{proto_string[1]}', result[i][j]/size)



def linear_hamming_distance():
    #sort by image name, make comparisons, result to file, calculate averages, delete file(s)
    pass