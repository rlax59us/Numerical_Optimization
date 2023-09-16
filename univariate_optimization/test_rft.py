from root_finding_techniques.newtons import newtons_method
from root_finding_techniques.secant import secant_method
from root_finding_techniques.regular_falsi import regular_falsi_method
from root_finding_techniques.bisection import bisection_method
from visualization import rft_visualization
import numpy as np
import random

def problem1():
    f = lambda x: x**3 - x**2 - 15
    df = lambda x: 3*x**2 - 2*x
    
    return f, df

def problem2():
    f = lambda x: x**2 - 15
    df = lambda x: 2*x

    return f, df

def problem3():
    f = lambda x: np.arctan(0.1*x)
    df = lambda x: 0.1*(1+(0.1*x)**2)**(-1)

    return f, df

def problem4():
    f = lambda x: x**3 + np.cos(x)
    df = lambda x: 3*x**2 - np.sin(x)

    return f, df

if __name__ == "__main__":
    problems = [problem1(), problem2(), problem3()]

    for i, problem in enumerate(problems):
        print("Problem "+str(i+1))
        f, df = problem
        rft_visualization(f=f, xn=[], fxn=[], problem=str(i+1), type='graph')
        #Newton's method
        results = newtons_method(f=f, df=df, x0=10, epsilon=1e-10, max_iter=10000)
        rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(i+1), type='newtons')
        #secant method
        results = secant_method(f=f, x0=10, x1=random.randrange(9,10), epsilon=1e-10, max_iter=10000)
        rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(i+1), type='secant')
        #regular falsi method
        if i == 1:
            results = regular_falsi_method(f=f, a=0, b=10, epsilon=1e-10, max_iter=10000)
        else:
            results = regular_falsi_method(f=f, a=-5, b=10, epsilon=1e-10, max_iter=10000)
        rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(i+1), type='regular_falsi')
        #bisection method
        if i == 1:
            results = bisection_method(f=f, a=0, b=10, epsilon=1e-10, max_iter=10000)
        else:
            results = bisection_method(f=f, a=-5, b=10, epsilon=1e-10, max_iter=10000)
        rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(i+1), type='bisection')
        
    print("Problem 4")
    f, df = problem4()
    rft_visualization(f=f, xn=[], fxn=[], problem=str(4), type='graph')
    #secant method
    results = secant_method(f=f, x0=5, x1=random.randrange(4,5), epsilon=1e-10, max_iter=10000)
    rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(4), type='secant')
    #regular falsi method
    results = regular_falsi_method(f=f, a=-1, b=5, epsilon=1e-10, max_iter=10000)
    rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(4), type='regular_falsi_1_', range=2)
    results = regular_falsi_method(f=f, a=-2, b=5, epsilon=1e-10, max_iter=10000)
    rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(4), type='regular_falsi_2_', range=2)
    results = regular_falsi_method(f=f, a=-3, b=5, epsilon=1e-10, max_iter=10000)
    rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(4), type='regular_falsi_3_', range=2)
    results = regular_falsi_method(f=f, a=-4, b=5, epsilon=1e-10, max_iter=10000)
    rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(4), type='regular_falsi_4_', range=2)
    results = regular_falsi_method(f=f, a=-5, b=5, epsilon=1e-10, max_iter=10000)
    rft_visualization(f=f, xn=results[1], fxn=results[2], problem=str(4), type='regular_falsi_5_', range=2)
