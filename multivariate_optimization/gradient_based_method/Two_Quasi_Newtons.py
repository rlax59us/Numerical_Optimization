import numpy as np

def SR1(Bk, yk, sk):
    r=1e-8
    if np.dot(np.transpose(yk - np.dot(Bk, sk)), sk) >= r*np.linalg.norm(sk)*np.linalg.norm(yk-np.dot(Bk, sk)):
        return Bk + (yk - Bk*sk)*np.transpose(yk-Bk*sk) / np.transpose(yk-Bk*sk)*sk
    else:
        return Bk

def BFGS(Bk, yk, sk):
    if np.dot(np.transpose(sk), yk) > 0:
        return Bk + (yk - Bk*sk)*np.transpose(yk-Bk*sk) / np.transpose(yk-Bk*sk)*sk
    else:
        return Bk

def Quasi_newtons(f, df, d2f, initial, ak=5e-5, max_iter=100000, type='SR1' ,threshold=1e-15):
    i = 0
    x = [initial]
    b = []
    fvalues = [f(initial[0], initial[1])]

    try:
        hessian = d2f(x[i][0], x[i][1])
    except:
        hessian = d2f
    b.append(np.linalg.inv(hessian))

    while(1):
        gradient = df(x[i][0], x[i][1])
        magnitude = np.linalg.norm(np.dot(b[i], gradient))
        pk = -np.dot(b[i], gradient) / magnitude
        x.append((x[i][0] + ak * pk[0], x[i][1] + ak * pk[1]))
        fvalues.append(f(x[i+1][0], x[i+1][1]))
        gradient_after = df(x[i+1][0], x[i+1][1])
        if type == 'SR1':
            Bk=np.array(b[i]) 
            yk=np.array(np.array(gradient_after)-np.array(gradient))
            sk=np.array(ak*pk)
            b.append(np.ndarray.tolist(SR1(Bk=Bk, yk=yk, sk=sk)))
        else:
            Bk=np.array(b[i]) 
            yk=np.array(np.array(gradient_after)-np.array(gradient))
            sk=np.array(np.array(x[i+1])-np.array(x[i]))
            b.append(np.ndarray.tolist(BFGS(Bk=Bk, yk=yk, sk=sk)))
        i += 1

        if fvalues[i-1] - fvalues[i] <= threshold:
            return x, fvalues
        if i >= max_iter:
            return x, fvalues