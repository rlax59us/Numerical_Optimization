from gradient_based_method.Steepest_Descent import Steepest_Descent
from gradient_based_method.Newtons import Newtons
from visualization import problem_visualization, gbm_visualization
import random

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

if __name__ == "__main__":
    problems = [problem1(), problem2(), problem3()]


    for number, problem in enumerate(problems):
        print("Problem "+str(number+1))
        f, df, d2f = problem
        given_initial_point = (1.2, 1.2)
        random_initial_point = (random.random()*(-2 if random.random() > 0.5 else 2), random.random()*(-2 if random.random() > 0.5 else 2))
        problem_visualization(f=f, problem=str(number+1), method='gbm', r=2)
        results = Steepest_Descent(f=f, df=df, d2f=d2f, initial=given_initial_point)
        gbm_visualization(f=f, results=results, problem=str(number+1), type='steepes_descent', r=5)
        results = Newtons(f=f, df=df, d2f=d2f, initial=given_initial_point)
        gbm_visualization(f=f, results=results, problem=str(number+1), type='newtons', r=5)

