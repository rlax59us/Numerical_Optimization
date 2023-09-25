import random

def triangle_area(a, b, c):
    #Heron's Formula
    s = (a+b+c)/2
    return (s*(s-a)*(s-b)*(s-c))**(0.5)

def reinitialize(dim):
    initial_point = []
    for i in range(dim + 1):
        initial_point.append((random.randrange(5, 10), random.randrange(5, 10)))
    return initial_point

def Nelder_Mead(f, initial, dim=2, alpha=1, beta=3, gamma=0.5, max_iter=1000, threshold=1e-15):
    #init
    print("-------init-------")
    point_list = initial
    iter = 0
    candidate = []
    p_best = []

    for point in point_list:
        candidate.append((point, f(point[0], point[1])))
    
    while(1):
        p_best.append(candidate[0][0])
        iter += 1
        #stop condition
        xr, yr = 0, 0
        for i in range(dim+1):
            if i < dim:
                xr+=candidate[i][0][0]*candidate[i+1][0][1]
                yr+=candidate[i][0][1]*candidate[i+1][0][0]
            else:
                xr+=candidate[i][0][0]*candidate[0][0][1]
                yr+=candidate[i][0][1]*candidate[0][0][0]
        
        area = abs((xr-yr)/2)

        if area < threshold:
            if iter == 1:
                #fail to find search direction
                reinitial = reinitialize(dim=dim)
                #retry
                return Nelder_Mead(f=f, initial=reinitial, dim=dim)
            return p_best
        if i > max_iter:
            return p_best
        
        #Reflection
        candidate = sorted(candidate, key = lambda item: item[-1])

        centroid = (0, 0)
        for item in candidate[:-1]:
            point = item[0]
            centroid = [(centroid[i] + point[i])/dim for i in range(len(point))]
        
        x_r = [centroid[i] + alpha*(centroid[i] - candidate[-1][0][i]) for i in range(len(centroid))]
        f_r = f(x_r[0], x_r[1])
        f_1 = candidate[0][1]
        f_n = candidate[-2][1]

        if f_1 <= f_r and f_r <= f_n:
            candidate = candidate[:-1] + [(x_r, f_r)]
        
        if f_r >= f_n:
            #contraction
            x_n1 = candidate[-1][0]
            f_n1 = candidate[-1][1]
            x_c = None
            if f_r < f_n1:
                x_c = [centroid[i] + gamma*(x_r[i] - centroid[i]) for i in range(len(centroid))]
            else:
                x_c = [centroid[i] + gamma*(x_n1[i] - centroid[i]) for i in range(len(centroid))]
            
            f_c = f(x_c[0], x_c[1])

            if f_c < min(f_r, f_n1):
                candidate = candidate[:-1] + [(x_c, f_c)]
            else:
                x_1 = candidate[0][0]
                new_candidate = [candidate[0]]
                for i in range(1, dim+1):
                    x_i = candidate[i][0]
                    new_x_i = [(x_i[i] + x_1[i])/2 for i in range(len(x_i))]
                    new_f_i = f(new_x_i[0], new_x_i[1])
                    new_candidate.append((new_x_i, new_f_i))
                candidate = new_candidate

        if f_r <= f_1:
            #expansion
            x_e = [centroid[i] + beta*(x_r[i] - centroid[i]) for i in range(len(centroid))]
            f_e = f(x_e[0], x_e[1])
            if f_e <= f_r:
                candidate = candidate[:-1] + [(x_e, f_e)]
            else:
                candidate = candidate[:-1] + [(x_r, f_r)]