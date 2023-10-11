import numpy as np
import math

def convergence_check(f, x, x_op, r):
    return 0

def Steepest_Descent(f, df, d2f, initial, ak=5e-2, dim=2, max_iter=1000, threshold=1e-15):
    i = 0
    x = [initial]

    while(1):
        direction = df(x[i][0],x[i][1])
        magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
        pk = (-direction[0]/magnitude, -direction[1]/magnitude) #unit direction
        x.append((x[i][0] + ak * pk[0], x[i][1] + ak * pk[1]))
        i += 1
        try:
            hessian = d2f(x[i][0], x[i][1])
        except:
            hessian = d2f
        eigvalues, eigvectors = np.linalg.eig(np.array(hessian))
        r = np.min(eigvalues) / np.max(eigvalues) 
        if convergence_check(f=f, x=x[i-1], x_op=x[i], r=r):
            return x
        if i >= max_iter:
            return x