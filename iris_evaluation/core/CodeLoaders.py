import numpy as np
import torch
from torch.utils.data import Dataset

class CodeLoader(Dataset):
    def __init__(self, iris_codes):
        self.iris_codes = iris_codes
        self.names = list(iris_codes.files)
        self.names.sort()
        self.height = iris_codes[self.names[0]].shape[2] * iris_codes[self.names[0]].shape[1] #real + imaginary portions
        self.width = iris_codes[self.names[0]].shape[3]

    def __len__(self):
        return len(self.names)

class RolledCodes(CodeLoader):
    def __getitem__(self, idx):
        iris_code = torch.empty((self.width, self.height * self.width), dtype=torch.bool) #allocates memory
        for i in range(self.width):
            iris_code[i] = torch.from_numpy(np.roll(self.iris_codes[self.names[idx]], shift=i, axis=3).reshape(1, self.height*self.width)) #roll, flatten, store

        return iris_code, self.names[idx]
    
class UnrolledCodes(CodeLoader):
    def __getitem__(self, idx):
        iris_code = torch.from_numpy(self.iris_codes[self.names[idx]].reshape(-1))

        return iris_code, self.names[idx]
    
class LinearUnrolled(Dataset):
    def __init__(self, reference_codes, comparison_codes):
        self.reference_codes = reference_codes
        self.reference_names = list(self.reference_codes.files)
        self.reference_names.sort()
        
        self.comparison_codes = comparison_codes
        self.comparison_names = list(self.comparison_codes.files)
        self.comparison_names.sort()


        self.comparison_height = comparison_codes[self.names[0]].shape[2] * comparison_codes[self.names[0]].shape[1] #real + imaginary portions
        self.comparison_width = comparison_codes[self.names[0]].shape[3]

        self.condition = self.decode_condition()

    def __len__(self):
        return len(self.reference_names)

    def __getitem__(self, idx):
        comparison_code = torch.empty((self.width, self.height * self.width), dtype=torch.bool) #allocates memory
        for i in range(self.width):
            comparison_code[i] = torch.from_numpy(np.roll(self.comparison_codes[self.comparison_names[idx]], shift=i, axis=3).reshape(1, self.comparison_height*self.comparison_width)) #roll, flatten, store

        comparison_code = torch.from_numpy(self.comparison_codes[self.comparison_names[idx]].reshape(-1))
        reference_code = torch.from_numpy(self.reference_codes[self.reference_names[idx]].reshape(-1))

        
        return (reference_code, self.reference_names[idx]), (comparison_code, self.comparison_names[idx])

    def __get_conditions(self):
        condition = None
        condition_rotation = None
        condition_mask = None
        
        try:
            condition = self.names[0].split('___cond___')[1]
            condition_parts = condition.split('_')

            condition_rotation = condition_parts[0]
            condition_mask = condition_parts[1]        
        except:
            pass
        
        condition_dict = {
            'condition': condition,
            'rotation' : condition_rotation,
            'mask' : condition_mask,
        }

        return condition_dict
    
    conditions = property(fget = __get_conditions)
