
#loads iris codes in a way that rolls them all the way around
class RolledCodes(Dataset):
    def __init__(self, iris_codes):
        self.__iris_codes = iris_codes
        self.__names = list(iris_codes.files)
        self.__names.sort()
        self.__height = iris_codes[self.__names[0]].shape[2] * iris_codes[self.__names[0]].shape[1] #real + imaginary portions
        self.__width = iris_codes[self.__names[0]].shape[3]

    def __len__(self):
        return len(self.__names)

    def __getitem__(self, idx):
        iris_code = torch.empty((self.__width, self.__height * self.__width), dtype=torch.bool) #allocates memory
        for i in range(self.__width):
            iris_code[i] = torch.from_numpy(np.roll(self.__iris_codes[self.__names[idx]], shift=i, axis=3).reshape(1, self.__height*self.__width)) #roll, flatten, store
        
        return iris_code, self.__names[idx]

#loads iris codes in a way that does not roll them
class UnrolledCodes(Dataset):
    def __init__(self, iris_codes):
        self.__iris_codes = iris_codes
        self.__names = list(iris_codes.files)
        self.__names.sort()

    def __len__(self):
        return len(self.__names)

    def __getitem__(self, idx):
        iris_code = torch.from_numpy(self.__iris_codes[self.__names[idx]].reshape(-1))

        return iris_code, self.__names[idx]













class LinearRolled(Dataset):
    def __init__(self, reference_codes, comparison_codes):
        self.__reference_codes = reference_codes
        self.__reference_names = list(self.__reference_codes.files)
        self.__reference_names.sort()
        
        self.__comparison_codes = comparison_codes
        self.__comparison_names = list(self.__comparison_codes.files)
        self.__comparison_names.sort()


        self.__comparison_height = comparison_codes[self.__names[0]].shape[2] * comparison_codes[self.__names[0]].shape[1] #real + imaginary portions
        self.__comparison_width = comparison_codes[self.__names[0]].shape[3]

        self.condition = self.__decode_condition()

    def __len__(self):
        return len(self.__reference_names)

    def __getitem__(self, idx):
        comparison_code = torch.empty((self.__width, self.__height * self.__width), dtype=torch.bool) #allocates memory
        for i in range(self.__width):
            comparison_code[i] = torch.from_numpy(np.roll(self.__comparison_codes[self.__comparison_names[idx]], shift=i, axis=3).reshape(1, self.__comparison_height*self.__comparison_width)) #roll, flatten, store

        reference_code = torch.from_numpy(self.__reference_codes[self.__reference_names[idx]].reshape(-1))

        
        return (reference_code, self.__reference_names[idx]), (comparison_code, self.__comparison_names[idx])

    def __decode_condition(self):
        condition = None
        condition_rotation = None
        condition_mask = None
        
        try:
            condition = self.__comparison_names[0].split('___cond___')[1]
            condition_parts = condition.split('_')

            condition_rotation = condition_parts[0]
            condition_mask = condition_parts[1]        
        except:
            pass
        
        comparison_dict = {
            'condition': condition,
            'rotation' : condition_rotation,
            'mask' : condition_mask,
        }

        return comparison_dict


class LinearUnrolled(Dataset):
    def __init__(self, reference_codes, comparison_codes):
        self.__reference_codes = reference_codes
        self.__reference_names = list(self.__reference_codes.files)
        self.__reference_names.sort()
        
        self.__comparison_codes = comparison_codes
        self.__comparison_names = list(self.__comparison_codes.files)
        self.__comparison_names.sort()


        self.__comparison_height = comparison_codes[self.__names[0]].shape[2] * comparison_codes[self.__names[0]].shape[1] #real + imaginary portions
        self.__comparison_width = comparison_codes[self.__names[0]].shape[3]

        self.condition = self.__decode_condition()

    def __len__(self):
        return len(self.__reference_names)

    def __getitem__(self, idx):
        comparison_code = torch.empty((self.__width, self.__height * self.__width), dtype=torch.bool) #allocates memory
        for i in range(self.__width):
            comparison_code[i] = torch.from_numpy(np.roll(self.__comparison_codes[self.__comparison_names[idx]], shift=i, axis=3).reshape(1, self.__comparison_height*self.__comparison_width)) #roll, flatten, store

        comparison_code = torch.from_numpy(self.__comparison_codes[self.__comparison_names[idx]].reshape(-1))
        reference_code = torch.from_numpy(self.__reference_codes[self.__reference_names[idx]].reshape(-1))

        
        return (reference_code, self.__reference_names[idx]), (comparison_code, self.__comparison_names[idx])

    def __decode_condition(self):
        condition = None
        condition_rotation = None
        condition_mask = None
        
        try:
            condition = self.__comparison_names[0].split('___cond___')[1]
            condition_parts = condition.split('_')

            condition_rotation = condition_parts[0]
            condition_mask = condition_parts[1]        
        except:
            pass
        
        comparison_dict = {
            'condition': condition,
            'rotation' : condition_rotation,
            'mask' : condition_mask,
        }

        return comparison_dict
