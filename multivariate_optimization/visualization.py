import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualization(f, results, problem='1', type='nelder_mead', r=10):
    point_x = []
    point_y = []
    for result in results:
        point_x.append(result[0])
        point_y.append(result[1])
    
    r_x = max(point_x) if max(point_x) > r else r
    r_y = max(point_y) if max(point_y) > r else r
        
    x = np.linspace(-r_x, r_x, 50)
    y = np.linspace(-r_y, r_y, 50)

    z = np.zeros((len(x), len(y)))
    
    for i in range(50):
        for j in range(50):
            z[i,j] = f(x[i], y[j])

    xx, yy = np.meshgrid(x, y)

    plt.figure(figsize=(5, 3.5))
    cp = plt.contourf(xx, yy, z, levels = np.linspace(z.reshape(-1, 1).min(), z.reshape(-1, 1).max(), 50))
    plt.colorbar(cp)
    
    plt.plot(point_x, point_y, marker='o', markersize=2, color='r')
    plt.savefig('multivariate_optimization/results/' + type + 'problem' + problem + '.png')
    plt.cla()

def problem_visualization(f, problem='1', r=10):
    x = np.linspace(-r, r, 50)
    y = np.linspace(-r, r, 50)

    z = np.zeros((len(x), len(y)))
    
    for i in range(50):
        for j in range(50):
            z[i,j] = f(x[i], y[j])

    xx, yy = np.meshgrid(x, y)

    plt.figure(figsize=(5, 3.5))
    cp = plt.contourf(xx, yy, z, levels = np.linspace(z.reshape(-1, 1).min(), z.reshape(-1, 1).max(), 50))
    plt.colorbar(cp)
    plt.savefig('multivariate_optimization/results/' + 'problem' + problem + '.png')
    plt.cla()