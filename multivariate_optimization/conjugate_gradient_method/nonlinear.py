import numpy as np
import random
from conjugate_gradient_method.wolfe_condition import generate_alpha
'''
def check_wolfe(f, df, x, ak, pk):
    c1 = 1e-3
    c2 = 0.9
    xn = x + ak*pk
    if f(xn[0], xn[1]) <= np.linalg.norm(c1*ak*np.dot(np.array(df(x[0], x[1])), pk)) + f(x[0], x[1]) and np.abs(np.dot(df(xn[0], xn[1]),pk)) <= np.abs(c2*np.dot(df(x[0], x[1]), pk)):
        return True
    else:
        return False

def generate_alpha(f, df, xk, pk):
    random_value = 0.0
    shift = 1
    random_value = random.random()*1e-3
    while check_wolfe(f, df, xk, random_value, pk) == False:
        random_value *= 0.1
        shift += 1
        if shift > 9:
            random_value = random.random()*1e-3
            shift = 1
    
    return random_value
'''
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
        #alpha = generate_alpha(f, df, x[k], p[k])
        alpha = generate_alpha(func=f, jac=df, x_k=x[k], p_k=p[k].copy())
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