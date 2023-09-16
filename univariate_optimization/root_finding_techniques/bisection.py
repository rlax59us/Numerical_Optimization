def bisection_method(f, a, b, epsilon, max_iter):
    xn = None
    an = a
    bn = b

    xn_process = []
    fxn_process = []

    if f(a)*f(b) >= 0:
        return None, xn_process, fxn_process

    for n in range(0, max_iter):
        xn = (an + bn) / 2
        xn_process.append(xn)
        fxn = f(xn)
        fxn_process.append(fxn)
        
        if abs(fxn) < epsilon:
            print(n)
            return xn, xn_process, fxn_process
        
        if f(a) * fxn < 0:
            bn = xn
        elif f(b) * fxn < 0:
            an = xn

    return None, xn_process, fxn_process

