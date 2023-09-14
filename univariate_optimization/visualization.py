import matplotlib.pyplot as plt
import numpy as np

def visualization(f, xn, fxn, problem='1', type='newtons', range=10):
    x = np.linspace(-range, range, 201)
    
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    plt.grid(color='gray', alpha=.5, linestyle='--')

    plt.plot(x, f(x))
    plt.plot(xn, fxn, marker='o', markersize=5)

    plt.savefig('univariate_optimization/results/'+ type + 'problem' + problem + '.png')
    plt.cla()