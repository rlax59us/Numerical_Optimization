from conjugate_gradient_method.linear import LinearCG
from conjugate_gradient_method.nonlinear import NonLinearCG
from visualization import problem_visualization, cgm_visualization, cgm_function_value_visualization
import time

def problem1():
    f = lambda x, y: (x+3*y-5)**2 +(3*x+y-7)**2 
    f_A = [[1, 3], [3, 1]]
    f_b = [5, 7]

    return f, f_A, f_b   

def problem2():
    f = lambda x, y: 50*(x-y**2)**2+(1-y)**2
    df_x = lambda x, y: 100*(x-y**2)
    df_y = lambda x, y: (-2*y)*100*(x-y**2) - 2*(1-y)

    df = lambda x, y: (df_x(x, y), df_y(x, y))
    d2f = lambda x, y: ((100, -200*y), (-200*y, -200*x + 600*(y**2)+2))

    return f, df, d2f

def problem3():
    f = lambda x, y: (1.5-x+x*y)**2+(2.25-x+x*(y**2))**2+(2.625-x+x*(y**3))**2
    df_x = lambda x, y : 2*(y-1)*(1.5-x+x*y)+2*(-1+y**2)*(2.25-x+x*(y**2))+2*(-1+y**3)*(2.625-x+x*(y**3))
    df_y = lambda x, y : 2*x*(1.5-x+x*y)+ 2*(2*x*y)*(2.25-x+x*(y**2))+2*(3*x*(y**2))*(2.625-x+x*(y**3))

    df = lambda x, y: (df_x(x, y), df_y(x, y))
    dxdyf = lambda x, y: 4*x*(y**2 - 1)*y + 4*y*(x*(y**2) - x + 2.25) + 6*x*(y**3 - 1)*(y**2) + 6*(y**2)*(x*(y**3) - x + 2.625) + 2*x*(y - 1) + 2*(x*y - x + 1.5)
    d2f = lambda x, y: ((2*(y**3 - 1)**2 + 2*(y**2 - 1)**2 + 2*(y - 1)**2, dxdyf(x,y)), (dxdyf(x,y), 18*(x**2)*(y**4) + 8*(x**2)*(y**2) + 2*(x**2) + 12*x*y*(x*(y**3) - x + 2.625) + 4*x*(x*(y**2) - x + 2.25)))
    
    return f, df, d2f

if __name__ == "__main__":
    problems = [problem2(), problem3()]
    initial_point = (4, 4)
    print("Problem "+str(1))
    f, f_A, f_b = problem1()
    results, linear_fvalues = LinearCG(f=f, f_A=f_A, f_b=f_b, initial=initial_point)
    cgm_visualization(f=f, results=results, problem=str(1), type='linear', r=5)

    for number, problem in enumerate(problems):
        print("Problem "+str(number+2))
        f, df, d2f = problem
        print('FR')
        results, fr_fvalues = NonLinearCG(f=f, df=df, d2f=d2f, type='FR', initial=initial_point)
        cgm_visualization(f=f, results=results, problem=str(number+2), type='FR', r=5)
        print('PR')
        results, pr_fvalues = NonLinearCG(f=f, df=df, d2f=d2f, type='PR', initial=initial_point)
        cgm_visualization(f=f, results=results, problem=str(number+2), type='PR', r=5)
        print('HS')
        results, hs_fvalues = NonLinearCG(f=f, df=df, d2f=d2f, type='HS', initial=initial_point)
        cgm_visualization(f=f, results=results, problem=str(number+2), type='HS', r=5)
        cgm_function_value_visualization(fr=fr_fvalues, hs=hs_fvalues, pr=pr_fvalues, problem=str(number+2), type='')


        
        
        
        