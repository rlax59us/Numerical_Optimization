import matplotlib.pyplot as plt
import numpy as np

def rft_visualization(f, xn, fxn, problem='1', type='newtons', range=10):
    x = np.linspace(-range, range, 201)
    
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    plt.grid(color='gray', alpha=.5, linestyle='--')

    plt.plot(x, f(x))
    plt.plot(xn, fxn, marker='o', markersize=5)

    plt.savefig('univariate_optimization/results/rft/'+ type + 'problem' + problem + '.png')
    plt.cla()

def ufo_visualization(f, a, b, problem='1', type='fibonacci', range=10):
    x = np.linspace(-range, range, 201)
    
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    plt.grid(color='gray', alpha=.5, linestyle='--')

    plt.plot(x, f(x))
    
    fa = []
    fb = []
    for ea, eb in zip(a, b):
        fa.append(f(ea))
        fb.append(f(eb))
    
    plt.plot(a, fa, marker='o', markersize=5)
    plt.plot(b, fb, marker='v', markersize=5)

    plt.savefig('univariate_optimization/results/ufo/'+ type + 'problem' + problem + '.png')
    plt.cla()