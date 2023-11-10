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

def cgm_visualization(f, results, problem='1', type='linear', r=10):
    point_x = []
    point_y = []
    for result in results:
        point_x.append(result[0])
        point_y.append(result[1])
    
    if max(point_x) > r:
        r_x = max(point_x) + 0.5
    elif np.abs(min(point_x)) > r:
        r_x = np.abs(min(point_x)) + 0.5
    else: 
        r_x = r

    if max(point_y) > r:
        r_y = max(point_y) + 0.5
    elif np.abs(min(point_y)) > r:
        r_y = np.abs(min(point_y)) + 0.5
    else: 
        r_y = r
        
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
    plt.savefig('multivariate_optimization/results/cgm/' + type + 'problem' + problem + '.png')
    cb.remove()
    plt.cla()

def summarize(array, interval):
    x=[]
    y=[]
    for i in range(len(array)):
        if i % interval == 0:
            x.append(i)
            y.append(array[i])

    return x, y

def gbm_function_value_visualization(sd, nt, sr1, bfgs, problem='1', type=''):
    result_len = max(len(sd), len(nt), len(sr1), len(bfgs))
    x, y = summarize(sd, int(result_len/25))
    plt.plot(x, y, label='Steepest Descent', marker='o')
    x, y = summarize(nt, int(result_len/25))
    plt.plot(x, y, label='Newtons', marker='v')
    x, y = summarize(sr1, int(result_len/25))
    plt.plot(x, y, label='Quasi_SR1', marker='s')
    x, y = summarize(bfgs, int(result_len/25))
    plt.plot(x, y, label='Quasi_BFGS', marker='*')
    plt.legend()
    plt.savefig('multivariate_optimization/results/gbm/' + 'function_value_problem' + problem + type +'.png')
    plt.cla()

def split_parameter(parameter):
    a = []
    b = []
    c = []
    d = []
    for i in parameter:
        a.append(i[0])
        b.append(i[1])
        c.append(i[2])
        d.append(i[3])
    
    return a,b,c,d

def lsm_parameter_visualization(parameter, method='GN', data='data1', model='model1'):
    result_len = len(parameter)
    a, b, c, d = split_parameter(parameter)
    x, y = summarize(a, int(result_len/25))
    plt.plot(x, y, label='a', marker='o')
    x, y = summarize(b, int(result_len/25))
    plt.plot(x, y, label='b', marker='v')
    x, y = summarize(c, int(result_len/25))
    plt.plot(x, y, label='c', marker='s')
    x, y = summarize(d, int(result_len/25))
    plt.plot(x, y, label='d', marker='*')
    plt.legend()
    plt.savefig('multivariate_optimization/results/lsm/parameter_' + method + data + model +'.png')
    plt.cla()

def lsm_function_value_visualization(results, method='GN', data='data1', model='model1'):
    result_len = len(results)
    x, y = summarize(results, int(result_len/25))
    plt.plot(x, y, label='residual', marker='o', markersize=8)
    plt.legend()
    plt.savefig('multivariate_optimization/results/lsm/residual_'  + method + data + model +'.png')
    plt.cla()

def cgm_function_value_visualization(fr, hs, pr, problem='1', type=''):
    result_len = max(len(fr), len(hs), len(pr))
    x, y = summarize(fr, int(result_len/25))
    plt.plot(x, y, label='cg_fr', marker='o', markersize=8)
    x, y = summarize(hs, int(result_len/25))
    plt.plot(x, y, label='cg_hs', marker='v', markersize=8)
    x, y = summarize(pr, int(result_len/25))
    plt.plot(x, y, label='cg_pr', marker='s', markersize=4)
    plt.legend()
    plt.savefig('multivariate_optimization/results/cgm/' + 'function_value_problem' + problem + type +'.png')
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