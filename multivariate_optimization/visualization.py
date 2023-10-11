import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def nsf_visualization(f, results, problem='1', type='nelder_mead', r=10):
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
    plt.savefig('multivariate_optimization/results/nsf/' + type + 'problem' + problem + '.png')
    plt.cla()

def gbm_visualization(f, results, problem='1', type='nelder_mead', r=10):
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
    cb = plt.colorbar(cp)
    
    plt.plot(point_x, point_y, marker='o', markersize=2, color='r')
    plt.savefig('multivariate_optimization/results/gbm/' + type + 'problem' + problem + '.png')
    cb.remove()
    plt.cla()

def summarize(array):
    x=[]
    y=[]
    for i in range(len(array)):
        if i %10 == 0:
            x.append(i)
            y.append(array[i])

    return x, y

def gbm_function_value_visualization(sd, nt, sr1, bfgs, problem='1'):
    x, y = summarize(sd)
    plt.plot(x, y, label='Steepest Descent', marker='o')
    x, y = summarize(nt)
    plt.plot(x, y, label='Newtons', marker='v')
    x, y = summarize(sr1)
    plt.plot(x, y, label='Quasi_SR1', marker='s')
    x, y = summarize(bfgs)
    plt.plot(x, y, label='Quasi_BFGS', marker='*')
    plt.legend()
    plt.savefig('multivariate_optimization/results/gbm/' + 'function_value_problem' + problem + '.png')
    plt.cla()

def problem_visualization(f, problem='1', method = 'nsf', r=10):
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
    plt.savefig('multivariate_optimization/results/'+ method  + '/problem' + problem + '.png')
    plt.cla()