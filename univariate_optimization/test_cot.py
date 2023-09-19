from comparison_optimization_techniques.fibonacci import fibonacci_search
from comparison_optimization_techniques.golden_section import golden_section_search
from comparison_optimization_techniques.seeking_bound import seeking_bound
from visualization import ufo_visualization
import numpy as np
import time
import random

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
    trials = 1000
    f_interval = 0
    f_time = 0
    g_interval =0
    g_time = 0

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
        
        print("Repetition of Experiment")
        for i in range(trials):
            a,b = seeking_bound(f=f, x0=random.randrange(-35, 35), d0=1)
            f_start = time.time()
            f_results = fibonacci_search(f=f, a=a, b=b, max_iter=1000)
            f_end = time.time()
            g_start = time.time()
            g_results = golden_section_search(f=f, a=a, b=b, max_iter=1000)
            g_end = time.time()
            f_time += f_end - f_start
            f_interval += f_results[1][-1]-f_results[0][-1]
            g_time += g_end - g_start
            g_interval += g_results[1][-1]-g_results[0][-1]
        print(f_time / 1000)
        print(f_interval / 1000)
        print(g_time / 1000)
        print(g_interval / 1000)

        

