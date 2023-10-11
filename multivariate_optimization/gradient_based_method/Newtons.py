import numpy as np
import math

def Newtons(f, df, d2f, initial, ak=5e-2, dim=2, max_iter=1000, threshold=1e-15):
    i = 0
    x = [initial]

    while(1):
        try:
            hessian = d2f(x[i][0], x[i][1])
        except:
            hessian = d2f
        inverse_hessian = np.linalg.inv(hessian)
        gradient = df(x[i][0], x[i][1])
        pk = -np.dot(inverse_hessian, gradient)
        x.append((x[i][0] + ak * pk[0], x[i][1] + ak * pk[1]))
        i += 1
        
        if i >= max_iter:
            return x