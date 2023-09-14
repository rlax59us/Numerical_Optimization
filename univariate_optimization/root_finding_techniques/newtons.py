def newtons_method(f, df, x0, epsilon, max_iter):
    xn = x0
    xn_process = []
    fxn_process = []
    for n in range(0, max_iter):
        xn_process.append(xn)
        fxn = f(xn)
        fxn_process.append(fxn)
        if abs(fxn) < epsilon:
            print(n)
            return xn, xn_process, fxn_process
        dfxn = df(xn)
        if dfxn == 0:
            return None, xn_process, fxn_process
        xn = xn - fxn/dfxn

    return None, xn_process, fxn_process