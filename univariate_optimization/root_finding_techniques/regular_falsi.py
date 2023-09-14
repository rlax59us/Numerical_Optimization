def regular_falsi_method(f, a, b, epsilon, max_iter):
    xn = None
    an = a
    bn = b

    xn_process = []
    fxn_process = []

    if f(a)*f(b) >= 0:
        return None, xn_process, fxn_process

    for n in range(1, max_iter):
        xn = an - f(an)*(bn - an)/(f(bn) - f(an))
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