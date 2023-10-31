import numpy as np

def LinearCG(f, f_A, f_b, initial, alpha=5e-5, max_iter=100000,threshold=1e-10):
    k = 0
    x = [initial]
    fvalues = [f(initial[0], initial[1])]
    r = [np.dot(f_A, x[0]) - f_b]
    p = [-r[0]]
    
    while(np.linalg.norm(r[k]) != 0):
        alpha = np.dot(np.transpose(r[k]), r[k]) / np.dot(np.dot(np.transpose(p[k]), f_A), p[k])
        x.append(x[k] + alpha * p[k])
        fvalues.append(f(x[k+1][0], x[k+1][1]))
        r.append(r[k] + alpha*np.dot(f_A, p[k]))
        beta = np.dot(np.transpose(r[k+1]), r[k+1]) / np.dot(np.transpose(r[k]), r[k])
        p.append(-r[k+1] + beta*p[k])
        
        k += 1

        if fvalues[k-1] - fvalues[k] <= threshold:
            return x, fvalues
        if k >= max_iter:
            return x, fvalues
    