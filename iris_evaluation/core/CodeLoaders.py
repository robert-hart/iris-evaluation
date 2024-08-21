import numpy as np
import torch
from torch.utils.data import Dataset

#loads iris codes in a way that rolls them all the way around
class RolledCodes(Dataset):
    def __init__(self, iris_codes):
        self.iris_codes = iris_codes
        self.names = (iris_codes.files).sort()
        self.height = iris_codes[self.names[0]].shape[2] * iris_codes[self.names[0]].shape[1] #real + imaginary portions
        self.width = iris_codes[self.names[0]].shape[3]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        iris_code = torch.empty((self.width, self.height * self.width), dtype=torch.bool) #allocates memory
        for i in range(self.width):
            iris_code[i] = torch.from_numpy(np.roll(self.iris_codes[self.names[idx]], shift=i, axis=3).reshape(1, self.height*self.width)) #roll, flatten, store
        
        return iris_code, self.names[idx]

#loads iris codes in a way that does not roll them
class UnrolledCodes(Dataset):
    def __init__(self, iris_codes):
        self.iris_codes = iris_codes
        self.names = (iris_codes.files).sort()
        self.height = iris_codes[self.names[0]].shape[2] * iris_codes[self.names[0]].shape[1] #real + imaginary portions
        self.width = iris_codes[self.names[0]].shape[3]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        iris_code = torch.from_numpy(self.iris_codes[self.names[idx]].reshape(-1))

        return iris_code, self.names[idx]
    
#for linear iris code comparison
class LinearCodes(Dataset):
    def __init__(self, iris_codes):
        self.iris_codes = iris_codes
        self.names = iris_codes.files
        self.height = iris_codes[self.names[0]].shape[2] * iris_codes[self.names[0]].shape[1] #real + imaginary portions
        self.width = iris_codes[self.names[0]].shape[3]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        iris_code = torch.from_numpy(self.iris_codes[self.names[idx]].reshape(-1))

        return iris_code, self.names[idx]