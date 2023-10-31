#based on https://github.com/minrk/scipy-1/blob/master/scipy/optimize/linesearch.py
import numpy as np

def phi(f, xk, pk, alpha):
    x_kn = xk + alpha*pk
    value = f(x_kn[0], x_kn[1])

    return value
    
def der_phi(df, xk, pk, alpha):
    x_kn = xk + alpha*pk
    value = np.dot(df(x_kn[0], x_kn[1]), pk)
    
    return value

def zoom(alpha_i, alpha_in, f, df, xk, pk, c1, c2, max_iter=100, tol=1e-8):
    if alpha_i > alpha_in:
        temp = alpha_i
        alpha_i = alpha_in 
        alpha_in = temp

    j=0

    while j < max_iter:
        alpha_j = (alpha_in + alpha_i)/2

        phi_j = phi(f, xk, pk, alpha_j)
        der_phi_0 = der_phi(df, xk, pk, 0)
        phi_0 = phi(f, xk, pk, 0) 
        phi_lo = phi(f, xk, pk, alpha_i)

        if abs(alpha_in - alpha_i) < tol:
            return alpha_j

        if phi_j > (phi_0 + c1 * alpha_j * der_phi_0) or phi_j >= phi_lo:
            alpha_in = alpha_j
        else:
            der_phi_j = der_phi(df, xk, pk, alpha_j)
            
            if abs(der_phi_j) <= -c2*der_phi_0:
                return alpha_j

            if der_phi_j * (alpha_in - alpha_i) >= 0:
                alpha_in = alpha_i
            
            alpha_i = alpha_j
        j+=1
    
    return alpha_j #converge failed

def generate_alpha(f, df, xk, pk, max_value=1e-3, c1=1e-3, c2=0.9, max_iter=1000):
    alpha = [0, 0.9*max_value]
    i = 0

    while i < max_iter:
        xk = np.array(xk)

        phi_in = phi(f,xk,pk,alpha[i+1])
        phi_i = phi(f,xk,pk,alpha[i])
        der_phi_0 = der_phi(df,xk,pk, 0)
        phi_0 = phi(f,xk,pk, 0) + c1 * alpha[i+1] * der_phi_0
        
        if phi_in > phi_0 or (phi_in >= phi_i and i > 0):
            alpha_star = zoom(alpha[i],alpha[i+1],f, df, xk, pk, c1, c2)
            return alpha_star
        der_phi_in = der_phi(df,xk,pk, alpha[i+1])
        
        if abs(der_phi_in) <= -c2*der_phi_0:
            alpha_star = alpha[i+1]
            return alpha_star

        if der_phi_in >= 0:
            alpha_star = zoom(alpha[i+1],alpha[i],f, df, xk, pk, c1, c2)
            return alpha_star

        alpha.append(min(2*alpha[i+1], max_value))
        i+=1

    return 0 #converge failed

