import numpy as np

def Newtons(f, df, d2f, initial, ak=5e-2, max_iter=100000, threshold=1e-20):
    i = 0
    x = [initial]
    fvalues = [f(initial[0], initial[1])]

    while(1):
        try:
            hessian = d2f(x[i][0], x[i][1])
        except:
            hessian = d2f
        inverse_hessian = np.linalg.inv(hessian)
        gradient = df(x[i][0], x[i][1])
        pk = -np.dot(inverse_hessian, gradient)
        magnitude = np.linalg.norm(pk)
        x.append((x[i][0] + ak * pk[0]/magnitude, x[i][1] + ak * pk[1]/magnitude))
        fvalues.append(f(x[i+1][0], x[i+1][1]))
        i += 1
        if fvalues[i-1] - fvalues[i] <=threshold:
            return x, fvalues
        if i >= max_iter:
            return x, fvalues