import math

def golden_section_search(f, a, b, max_iter):
    ratio = (math.sqrt(5)-1)/2

    an = None
    bn = None

    a_process = []
    b_process = []

    for n in range(0, max_iter):
        a_process.append(a)
        b_process.append(b)

        L = b - a

        bn = a + ratio*L
        an = b - ratio*L

        fan = f(an)
        fbn = f(bn)

        if fan > fbn:
            a = an
        else:
            b = bn
    
    return a_process, b_process
