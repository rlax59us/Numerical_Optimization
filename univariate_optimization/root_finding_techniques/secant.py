def secant_method(f, x0, x1, epsilon, max_iter):
    xn_1 = x0
    xn = x1
    xn_process = []
    fxn_process = []

    for n in range(0, max_iter):
        xn_process.append(xn)
        fxn = f(xn)
        fxn_process.append(fxn)
        if abs(fxn) < epsilon:
            print(n)
            return xn, xn_process, fxn_process
        dfxn = (f(xn) - f(xn_1))/(xn - xn_1)
        if dfxn == 0:
            return None, xn_process, fxn_process
        xn_1 = xn
        xn = xn - fxn/dfxn

    return None, xn_process, fxn_process

