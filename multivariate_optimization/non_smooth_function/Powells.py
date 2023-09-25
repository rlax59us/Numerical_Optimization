import sys
sys.path.append('C:/Users/user/Desktop/Numerical_Optimization')

from univariate_optimization.comparison_optimization_techniques.golden_section import golden_section_search
from univariate_optimization.comparison_optimization_techniques.seeking_bound import seeking_bound
import numpy as np
import math

def get_function(f, p, u):
    new_f = lambda r: f(p[0] + r*u[0], p[1] + r*u[1])

    return new_f

def check_consecutive_estimates(f, x_k, x_k_1, threshold):
    difference = math.sqrt((x_k[0] - x_k_1[0])**2 + (x_k[1] - x_k_1[1])**2)
    if abs(difference) < threshold: #1
        return True
    elif abs(difference)/abs(math.sqrt(x_k[0]**2 + x_k[1]**2)) < threshold: #2
        return True
    elif abs(f(x_k[0], x_k[1])) == 0: #exception for 0(global minimum of f)
        return True
    elif abs(f(x_k[0], x_k[1]) - f(x_k_1[0], x_k_1[1]))/abs(f(x_k[0], x_k[1])) < threshold: #4
        return True
    return False

def Powells(f, df, initial, dim=2, max_iter=1000, threshold=1e-15):
    i = 0
    x = [initial]
    u = np.identity(dim).tolist()
    
    while(1):
        p = [x[i]]
        for k in range(0,dim):
            new_f = get_function(f, p[k], u[k])
            a, b = seeking_bound(new_f, 0.0, 0.1)
            results = golden_section_search(new_f, a, b, max_iter=1000)
            x1, x2 = results[0][-1], results[1][-1]
            f1, f2 = new_f(x1), new_f(x2)
            r = x1 if f1 < f2 else x2
            p.append((p[k][0]+r*u[k][0], p[k][1]+r*u[k][1]))

        i += 1
        
        for j in range(0, dim-1):
            u[j] = u[j+1]

        u[dim-1] = [p[dim][0] - p[0][0], p[dim][1] - p[0][1]]

        new_f = get_function(f, p[0], u[dim-1])
        a, b = seeking_bound(new_f, 0.0, 0.1)
        results = golden_section_search(new_f, a,b, max_iter=1000)
        x1, x2 = results[0][-1], results[1][-1]
        f1, f2 = new_f(x1), new_f(x2)
        r = x1 if f1 < f2 else x2
        x.append([p[0][0] + r*u[dim-1][0], p[0][1] + r*u[dim-1][1]])
        
        df_x_k = df(x[-1][0], x[-1][1])
        pk = p[-1]

        if i >= max_iter: #6
            return x
        if i > 1:
            if check_consecutive_estimates(f=f, x_k=x[-1], x_k_1=x[-2], threshold=threshold): #1,2,4
                return x
        if math.sqrt(df_x_k[0]**2 + df_x_k[1]**2) < threshold: #3
            return x
        if pk[0]*df_x_k[0] + pk[1]*df_x_k[1] <= 0: #5
            return x