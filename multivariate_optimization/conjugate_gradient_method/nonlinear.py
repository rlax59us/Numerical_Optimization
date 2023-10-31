import numpy as np
from conjugate_gradient_method.wolfe_condition import generate_alpha

def FR(df_k, df_kn):
    return np.dot(np.transpose(df_kn), df_kn) / np.dot(np.transpose(df_k), df_k)

def PR(df_k, df_kn):
    return np.dot(np.transpose(df_kn), df_kn - df_k) / np.dot(np.transpose(df_k), df_k)

def HS(df_k, df_kn, pk):
    return np.dot(np.transpose(df_kn), df_kn - df_k) / np.dot(np.transpose(df_kn - df_k), pk)

def NonLinearCG(f, df, d2f, initial, max_iter=100, type='FR' ,threshold=1e-15):
    k = 0
    x = [initial]
    fvalues = [f(initial[0], initial[1])]
    p = [-1*np.array(df(initial[0], initial[1]))]
    
    while(df(x[k][0], x[k][1]) != 0):
        alpha = generate_alpha(f=f, df=df, xk=x[k], pk=p[k].copy())
        x.append(x[k] + alpha*p[k])
        fvalues.append(f(x[k+1][0], x[k+1][1]))

        if type == 'FR':
            beta = FR(df_k=np.array(df(x[k][0], x[k][1])), df_kn=np.array(df(x[k+1][0], x[k+1][1])))
        elif type == 'PR':
            beta = PR(df_k=np.array(df(x[k][0], x[k][1])), df_kn=np.array(df(x[k+1][0], x[k+1][1])))
        else:
            beta = HS(df_k=np.array(df(x[k][0], x[k][1])), df_kn=np.array(df(x[k+1][0], x[k+1][1])), pk=p[k])
        p.append(-1 * np.array(df(x[k+1][0], x[k+1][1])) + beta*p[k])
        
        k += 1

        if fvalues[k-1] - fvalues[k] <= threshold:
            return x, fvalues
        if k >= max_iter:
            return x, fvalues
        
