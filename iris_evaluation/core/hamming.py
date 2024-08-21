from .CodeLoaders import RolledCodes, UnrolledCodes

from tqdm import tqdm
import torch

def set_torch_device():
    device = 'cpu'
    if torch.cuda.is_available(): #check if NVIDIA gpu is availible
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        print('No GPU or Apple Silicon availible. Operations may be slow.')
    return device

def append_text_to_file(file_path, comparison, result):
    with open(file_path, 'a') as file:
        file.write(f"{comparison}\t{result}\n")

def permutate_data(rolled, device, sample_batch, reference_batch_device):
    sample_batch_device = sample_batch[0].to(device)
    print(sample_batch_device.shape)

    result = None
    if rolled:
        sample_batch_device = sample_batch_device.unsqueeze(1)
        print(reference_batch_device.shape)
        print(sample_batch_device.shape)
        result = torch.bitwise_xor(reference_batch_device, sample_batch_device).sum(dim=-1).min(dim=-1)[0]
    else:
        sample_batch_device = sample_batch_device.unsqueeze(0)
        print(reference_batch_device.shape)
        print(sample_batch_device.shape)
        result = torch.bitwise_xor(reference_batch_device, sample_batch_device).sum(dim=-1)

    return result

#TODO create main HD function in here?

# take instructions, determine which function to use


#TODO make this an if/then

def pairwise_hamming_distance(device, cache_path, reference_codes, sample_codes, rolled):
    for reference_batch in tqdm(reference_codes):
        size = reference_batch[0].shape[-1]
        reference_batch_gpu = reference_batch[0].to(device)
        reference_batch_gpu = reference_batch_gpu.unsqueeze(1)
        if rolled:
            for sample_batch in sample_codes:
                sample_batch_gpu = sample_batch[0].to(device)
                sample_batch_gpu = sample_batch_gpu.unsqueeze(1)
                result = torch.bitwise_xor(reference_batch_gpu, sample_batch_gpu).sum(dim=-1).min(dim=-1)[0]
                for i in range(len(reference_batch[1])):
                    for j in range(len(sample_batch[1])):
                        proto_string = [reference_batch[1][i], sample_batch[1][j]]
                        proto_string.sort()
                        append_text_to_file(cache_path, f'{proto_string[0]}|{proto_string[1]}', result[i][j]/size)
        else:
            for sample_batch in sample_codes:
                sample_batch_gpu = sample_batch[0].to(device)
                sample_batch_gpu = sample_batch_gpu.unsqueeze(0)
                result = torch.bitwise_xor(reference_batch_gpu, sample_batch_gpu).sum(dim=-1)
                for i in range(len(reference_batch[1])):
                    for j in range(len(sample_batch[1])):
                        proto_string = [reference_batch[1][i], sample_batch[1][j]]
                        proto_string.sort()
                        append_text_to_file(cache_path, f'{proto_string[0]}|{proto_string[1]}', result[i][j]/size)


"""
def calculate_pairwise_hamming(device, cache_path, reference_codes, sample_codes):
    rolled = None

    if type(reference_codes) is RolledCodes:
        print(type(reference_codes))
        rolled = True
    else:
        rolled = False

    for reference_batch in tqdm(reference_codes):
        size = reference_batch[0].shape[-1]
        reference_batch_device = reference_batch[0].to(device)
        reference_batch_device = reference_batch_device.unsqueeze(1)
        print(reference_batch_device.shape)


        for sample_batch in sample_codes:
            result = permutate_data(rolled, device, sample_batch, reference_batch_device)

            for i in range(len(reference_batch[1])):
                for j in range(len(sample_batch[1])):
                    proto_string = [reference_batch[1][i], sample_batch[1][j]]
                    proto_string.sort()
                    append_text_to_file(cache_path, f'{proto_string[0]}|{proto_string[1]}', result[i][j]/size)
"""


def linear_hamming_distance():
    pass



#TODO create main HD function in here?