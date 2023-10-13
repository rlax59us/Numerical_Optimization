import numpy as np

def Steepest_Descent(f, df, d2f, initial, ak=5e-5, max_iter=100000, threshold=1e-15):
    i = 0
    x = [initial]
    fvalues = [f(initial[0], initial[1])]

    while(1):
        direction = df(x[i][0], x[i][1])
        magnitude = np.linalg.norm(direction)
        pk = -np.array(direction) / magnitude
        x.append((x[i][0] + ak * pk[0], x[i][1] + ak * pk[1]))
        fvalues.append(f(x[i+1][0], x[i+1][1]))
        i += 1
        if fvalues[i-1] - fvalues[i] <=threshold:
            return x, fvalues
        if i >= max_iter:
            return x, fvalues