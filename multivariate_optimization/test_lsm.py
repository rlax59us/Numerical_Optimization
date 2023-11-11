import math
from least_square_methods.dataloader import model_data
from least_square_methods.gauss_newton import gauss_newton
from least_square_methods.levenberg_marquardt import levenberg_marquardt
from visualization import *
import numpy as np

class model1():
    def __init__(self):
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 1
    
    def get_position(self):
        return np.array([self.a, self.b, self.c, self.d])
    
    def set_position(self, new_posiiton):
        self.a = new_posiiton[0]
        self.b = new_posiiton[1]
        self.c = new_posiiton[2]
        self.d = new_posiiton[3]

    def jacobian(self, x, y, z):
        return np.array([x, y, z, 1])

    def forward(self, x, y, z):
        return self.a*x + self.b*y + self.c*z + self.d
    
class model2():
    def __init__(self):
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 1

    def get_position(self):
        return np.array([self.a, self.b, self.c, self.d])
    
    def set_position(self, new_posiiton):
        self.a = new_posiiton[0]
        self.b = new_posiiton[1]
        self.c = new_posiiton[2]
        self.d = new_posiiton[3]

    def jacobian(self, x, y, z):
        element1 = (2/math.pow(self.d, 2))*(x - self.a)
        element2 = (2/math.pow(self.d, 2))*(y - self.b)
        element3 = (2/math.pow(self.d, 2))*(z - self.c)
        element4 = (2/math.pow(self.d, 3))*(math.pow(x-self.a, 2) + math.pow(y-self.b, 2) + math.pow(z-self.c, 2))

        return np.array([self.forward(x,y,z)*element1, self.forward(x,y,z)*element2, self.forward(x,y,z)*element3, self.forward(x,y,z)*element4])

    def gap(self, variable, value):
        return math.pow(variable - value, 2)

    def forward(self, x, y, z):
        return math.exp(-(self.gap(x, self.a) + self.gap(y, self.b) + self.gap(z, self.c))/math.pow(self.d, 2))

if __name__ == "__main__":
    
    #Gauss_newton
    first_model = model1()
    model1_data1 = model_data(data_type=1, file_name='Model1_Data.txt')
    track_parameter, results1 = gauss_newton(first_model, model1_data1)
    lsm_parameter_visualization(track_parameter, method='GN', data='data1', model='model1')

    second_model = model1()
    model1_data2 = model_data(data_type=2, file_name='Model1_Data.txt')
    track_parameter, results2 = gauss_newton(second_model, model1_data2)
    lsm_parameter_visualization(track_parameter, method='GN', data='data2', model='model1')

    lsm_function_value_visualization(results1, results2, method='GN', model='model1')
    
    third_model = model2()
    model2_data1 = model_data(data_type=1, file_name='Model2_Data.txt')
    track_parameter, results1 = gauss_newton(third_model, model2_data1)
    lsm_parameter_visualization(track_parameter, method='GN', data='data1', model='model2')
    
    fourth_model = model2()
    model2_data2 = model_data(data_type=2, file_name='Model2_Data.txt')
    track_parameter, results2 = gauss_newton(fourth_model, model2_data2)
    lsm_parameter_visualization(track_parameter, method='GN', data='data2', model='model2')

    lsm_function_value_visualization(results1, results2, method='GN', model='model2')
    
    #Levenberg_marquardt
    first_model = model1()
    model1_data1 = model_data(data_type=1, file_name='Model1_Data.txt')
    track_parameter, results1 = levenberg_marquardt(first_model, model1_data1)
    lsm_parameter_visualization(track_parameter, method='LM', data='data1', model='model1')

    second_model = model1()
    model1_data2 = model_data(data_type=2, file_name='Model1_Data.txt')
    track_parameter, results2 = levenberg_marquardt(second_model, model1_data2)
    lsm_parameter_visualization(track_parameter, method='LM', data='data2', model='model1')

    lsm_function_value_visualization(results1, results2, method='LM', model='model1')
    
    third_model = model2()
    model2_data1 = model_data(data_type=1, file_name='Model2_Data.txt')
    track_parameter, results1 = levenberg_marquardt(third_model, model2_data1)
    lsm_parameter_visualization(track_parameter, method='LM', data='data1', model='model2')
    
    fourth_model = model2()
    model2_data2 = model_data(data_type=2, file_name='Model2_Data.txt')
    track_parameter, results2 = levenberg_marquardt(fourth_model, model2_data2)
    lsm_parameter_visualization(track_parameter, method='LM', data='data2', model='model2')

    lsm_function_value_visualization(results1, results2, method='LM', model='model2')
