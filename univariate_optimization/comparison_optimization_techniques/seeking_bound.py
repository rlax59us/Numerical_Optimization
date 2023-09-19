def seeking_bound(f, x0, d0):
    k = 0
    x_left = x0 - d0
    x = x0
    x_right = x0 + d0
    
    while(True):
        f_left = f(x_left)
        f_middle = f(x)
        f_right = f(x_right)
        d = (2**k)*d0
        if f_left >= f_middle and f_middle >= f_right:
            x_left = x
            x = x_right
            x_right = x_right + d
        elif f_left <= f_middle and f_middle <= f_right:
            x_right = x
            x = x_left
            x_left = x_left - d
        else:
            break
        k += 1
    
    return x_left, x_right
