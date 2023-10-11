from gradient_based_method.Steepest_Descent import Steepest_Descent
from gradient_based_method.Newtons import Newtons
from gradient_based_method.Two_Quasi_Newtons import Quasi_newtons
from visualization import problem_visualization, gbm_visualization, gbm_function_value_visualization
import time

def problem1():
    f = lambda x, y: (x+3*y-5)**2 +(3*x+y-7)**2 
    df_x = lambda x, y: 2*(x+3*y-5) + 6*(3*x+y-7)
    df_y = lambda x, y: 6*(x+3*y-5) + 2*(3*x+y-7)

    df = lambda x, y: (df_x(x, y), df_y(x, y))
    d2f = ((20, 12), (12, 20))

    return f, df, d2f   

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

def do_experiment(f, df, d2f, number, initial_point):
    x, y = initial_point[0], initial_point[1]
    strxy = '_' + str(x) + '_' + str(y) +'_'
    start = time.time()
    results, sd_fvalues = Steepest_Descent(f=f, df=df, d2f=d2f, initial=initial_point)
    gbm_visualization(f=f, results=results, problem=str(number+1), type='steepest_descent'+strxy, r=5)
    end = time.time()
    print("SD: ", end-start)
    start = time.time()
    results, nt_fvalues = Newtons(f=f, df=df, d2f=d2f, initial=initial_point)
    gbm_visualization(f=f, results=results, problem=str(number+1), type='newtons'+strxy, r=5)
    end = time.time()
    print("NT: ", end-start)
    start = time.time()
    results, sr1_fvalues = Quasi_newtons(f=f, df=df, d2f=d2f, initial=initial_point, type='SR1')
    gbm_visualization(f=f, results=results, problem=str(number+1), type='quasi_newtons_sr1'+strxy, r=5)
    end = time.time()
    print("SR1: ", end-start)
    start = time.time()
    results, bfgs_fvalues = Quasi_newtons(f=f, df=df, d2f=d2f, initial=initial_point, type='BFGS')
    gbm_visualization(f=f, results=results, problem=str(number+1), type='quasi_newtons_bfgs'+strxy, r=5)
    end = time.time()
    print("BFGS: ", end-start)
    gbm_function_value_visualization(sd=sd_fvalues, nt=nt_fvalues, sr1=sr1_fvalues, bfgs=bfgs_fvalues, problem=str(number+1))

if __name__ == "__main__":
    problems = [problem1(), problem2(), problem3()]


    for number, problem in enumerate(problems):
        print("Problem "+str(number+1))
        f, df, d2f = problem
        given_initial_point = (1.2, 1.2)
        other_initial_point = (-2, -2)
        problem_visualization(f=f, problem=str(number+1), method='gbm', r=2)
        do_experiment(f=f, df=df, d2f=d2f, number=number, initial_point=given_initial_point)
        do_experiment(f=f, df=df, d2f=d2f, number=number, initial_point=other_initial_point)

        
        
        
        