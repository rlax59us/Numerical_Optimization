import pandas as pd
from torch.utils.data import Dataset

class model_data(Dataset):
    def __init__(self, data_type=1, file_name='Model1_Data.txt', path='C:/Users/user/Desktop/Numerical_Optimization/multivariate_optimization/least_square_methods/data/'):
        dataset = pd.read_table(path+file_name, sep="\t").drop([0], axis=0)
        if data_type==1:
            self.data = dataset['data1']
        else:
            self.data = dataset['data2']
        self.x = dataset['x']
        self.y = dataset['y']
        self.z = dataset['z']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.x[index%self.__len__()+1], self.y[index%self.__len__()+1], self.z[index%self.__len__()+1], self.data[index%self.__len__()+1]