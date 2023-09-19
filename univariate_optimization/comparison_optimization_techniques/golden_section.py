import math

def golden_section_search(f, a, b, max_iter):
    ratio = (math.sqrt(5)-1)/2

    a_process = []
    b_process = []

    bn = a + ratio*(b-a)
    an = b - ratio*(b-a)

    for n in range(0, max_iter):
        a_process.append(a)
        b_process.append(b)

        L = b - a
            
        if a == an:
            an = bn
            bn = a + ratio*L
        elif b == bn:
            bn = an
            an = b - ratio*L

        fan = f(an)
        fbn = f(bn)

        if fan > fbn:
            a = an
        else:
            b = bn
    
    return a_process, b_process
