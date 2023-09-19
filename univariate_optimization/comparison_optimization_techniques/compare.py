import math
def generate_fibonacci(n):
    fibo_list = [1, 1]
    for i in range(n-2):
        fibo_list.append(fibo_list[-1] + fibo_list[-2])
    
    return fibo_list

if __name__ =="__main__":
    ratio = (math.sqrt(5)-1)/2
    fibo_list = generate_fibonacci(1000)
    for i in range(3, 11):
        print('--------------------')
        fibo_ratio = fibo_list[i-2]/fibo_list[i]
        print(fibo_ratio)
        print(1-ratio)