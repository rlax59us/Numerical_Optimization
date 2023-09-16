from unimodal_function_optimization.fibonacci import fibonacci_search
from unimodal_function_optimization.golden_section import golden_section_search
from visualization import ufo_visualization
import numpy as np
import time

def problem1():
    f = lambda x: (x-10)**4 - 0.5*x**3 + (x+7)**2 - 15
    
    return f

def problem2():
    f = lambda x: (x+3)**4-0.5*x**2 + x + 6

    return f

def problem3():
    f = lambda x: 2*np.cos(x)

    return f

if __name__ == "__main__":
    problems = [problem1(), problem2(), problem3()]

    for i, problem in enumerate(problems):
        print("Problem "+str(i+1))
        f = problem
        ufo_visualization(f=f, a=[], b=[], problem=str(i+1), type='graph', range=40)
        #fibonacci
        f_start = time.time()
        results = fibonacci_search(f=f, a=-40, b=40, max_iter=1000)
        f_end = time.time()
        print(f"{f_end - f_start:.5f} sec")
        print(results[1][-1]-results[0][-1])
        ufo_visualization(f=f, a=results[0], b=results[1], problem=str(i+1), type='fibonacci', range=40)
        #golden section
        g_start = time.time()
        results = golden_section_search(f=f, a=-40, b=40, max_iter=1000)
        g_end = time.time()
        print(f"{g_end - g_start:.5f} sec")
        print(results[1][-1]-results[0][-1])
        ufo_visualization(f=f, a=results[0], b=results[1], problem=str(i+1), type='golden_section', range=40)
        

        
