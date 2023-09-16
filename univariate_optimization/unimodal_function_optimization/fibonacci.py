def generate_fibonacci(n):
    fibo_list = [1, 1]
    for i in range(n-2):
        fibo_list.append(fibo_list[-1] + fibo_list[-2])
    
    return fibo_list

def fibonacci_search(f, a, b, max_iter):
    fibo_list = generate_fibonacci(max_iter+2)

    a_process = []
    b_process = []

    for n in range(0, max_iter):
        a_process.append(a)
        b_process.append(b)

        L = b - a
        fibo = fibo_list[-(n+1)]
        fibo_1 = fibo_list[-(n+2)]
        fibo_2 = fibo_list[-(n+3)]

        an = a + (fibo_2/fibo)*L
        bn = b - (fibo_2/fibo)*L

        fan = f(an)
        fbn = f(bn)

        if fan > fbn:
            a = an
        else:
            b = bn
    
    return a_process, b_process

