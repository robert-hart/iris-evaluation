#optimized for parallel processing via GPU or Apple Silicon
from tqdm import tqdm
import torch

def set_device():
    device = 'cpu'
    if torch.cuda.is_available(): #check if NVIDIA gpu is availible
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        print('No GPU or Apple Silicon availible. May be slow.')
    return device

def append_text_to_file(file_path, comparison, result):
    with open(file_path, 'a') as file:
        file.write(f"{comparison}\t{result}\n")

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