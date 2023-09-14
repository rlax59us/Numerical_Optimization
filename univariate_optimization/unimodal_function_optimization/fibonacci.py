def fibonacci_search(f, a, b, max_iter):
    fibo_2 = 1
    fibo_1 = 1

    an = None
    bn = None

    a_process = []
    b_process = []

    for n in range(0, max_iter):
        a_process.append(a)
        b_process.append(b)

        L = b - a
        fibo = fibo_2 + fibo_1

        an = a + (fibo_2/fibo)*L
        bn = b - (fibo_2/fibo)*L

        fan = f(an)
        fbn = f(bn)

        if fan > fbn:
            a = an
        else:
            b = bn
        fibo_2 = fibo_1
        fibo_1 = fibo
    
    return a_process, b_process

