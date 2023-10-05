from non_smooth_function.Nelder_Mead import Nelder_Mead
from non_smooth_function.Powells import Powells
from visualization import visualization, problem_visualization
import random
import time

def problem1():
    f = lambda x, y: (x+3*y-5)**2 +(3*x+y-7)**2 
    df_x = lambda x, y: 2*(x+3*y-5) + 6*(3*x+y-7)
    df_y = lambda x, y: 6*(x+3*y-5) + 2*(3*x+y-7)

    df = lambda x, y: (df_x(x, y), df_y(x, y))

    return f, df

def problem2():
    f = lambda x, y: 50*(x-y**2)**2+(1-y)**2
    df_x = lambda x, y: 100*(x-y**2)
    df_y = lambda x, y: (-2*y)*100*(x-y**2) - 2*(1-y)

    df = lambda x, y: (df_x(x, y), df_y(x, y))

    return f, df

def problem3():
    f = lambda x, y: (1.5-x+x*y)**2+(2.25-x+x*(y**2))**2+(2.625-x+x*(y**3))**2
    df_x = lambda x, y : 2*(y-1)*(1.5-x+x*y)+2*(-1+y**2)*(2.25-x+x*(y**2))+2*(-1+y**3)*(2.625-x+x*(y**3))
    df_y = lambda x, y : 2*x*(1.5-x+x*y)+ 2*(2*x*y)*(2.25-x+x*(y**2))+2*(3*x*(y**2))*(2.625-x+x*(y**3))

    df = lambda x, y: (df_x(x, y), df_y(x, y))

    return f, df

if __name__ == "__main__":
    problems = [problem1(), problem2(), problem3()]
    dim = 2

    for number, problem in enumerate(problems):
        print("Problem "+str(number+1))
        f, df = problem
        problem_visualization(f=f, problem=str(number+1), r=100)

        initial_point = []
        for i in range(dim + 1):
            initial_point.append((random.randrange(50, 100)*(-1 if random.random() > 0.5 else 1), random.randrange(50, 100)*(-1 if random.random() > 0.5 else 1)))
        n_start = time.time()
        results = Nelder_Mead(f=f, initial=initial_point, dim=dim)
        n_end = time.time()
        print(f"{n_end - n_start:.5f} sec")
        visualization(f=f, results=results, problem=str(number+1), type='nelder_mead', r=100)
        p_start = time.time()
        results = Powells(f=f, df=df, initial=(initial_point[0][0], initial_point[0][1]), dim=dim)
        p_end = time.time()
        print(f"{p_end - p_start:.5f} sec")
        visualization(f=f, results=results, problem=str(number+1), type='powells', r=100)