from unimodal_function_optimization.fibonacci import fibonacci_search
from unimodal_function_optimization.golden_section import golden_section_search
from visualization import ufo_visualization
import numpy as np

def problem1():
    f = lambda x: (x-10)**4 - 0.5*x**3 + (x+7)**2 - 15
    
    return f

def problem2():
    f = lambda x: (x+3)**4-0.5*x**2 + x + 6

    return f

def problem3():
    f = lambda x: 2*x**4 + 2*(x-10)**3 + 8 

    return f

if __name__ == "__main__":
    problems = [problem1(), problem2(), problem3()]

    for i, problem in enumerate(problems):
        print("Problem "+str(i+1))
        f = problem

        results = fibonacci_search(f=f, a=-40, b=40, max_iter=10)
        ufo_visualization(f=f, a=results[0], b=results[1], problem=str(i+1), type='fibonacci', range=40)

        results = golden_section_search(f=f, a=-40, b=40, max_iter=10)
        ufo_visualization(f=f, a=results[0], b=results[1], problem=str(i+1), type='golden_section', range=40)
        

        
