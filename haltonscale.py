import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def first_primes(n):
    "List first n primes"
    prime_list = [2]
    num = 3
    while len(prime_list) < n:
        for p in prime_list:

            if num % p == 0:
                break
        else:
            prime_list.append(num)

        num += 2
    return prime_list

def halton(b):
    """Generator function for Halton sequence."""
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d  

def halton_nD(n, low=0, high=1, d=1):
    """Generate first n points in 2D-halton sequence, optional rescaling"""
    varnames = [f"x{i}" for i in range(1, d+1)]
    generators = [halton(i) for i in first_primes(d)]
    samples = np.array(list(zip(range(n), *generators)))[:, 1:] # drop index 
    df = pd.DataFrame(samples, columns=varnames)
    for var in varnames:
        df[var] = df[var]*(high - low) + low             
    return df   



if __name__=="__main__":
    N = 10000
    d = 2
    low = -1
    high = 1
    df = halton_nD(N, low=low, high=high, d=d)
    
    np.random.seed(123)
    r = np.random.uniform(low, high, size=(N, 2))
    size = 2
    plt.scatter(r[:, 0], r[:, 1], s=size, c="orange")
    plt.title("Random uniform")
    plt.savefig(f"figures/Random-{N}-samples.png")
    plt.show()
    plt.scatter(df.x1, df.x2, s=size, c="blue")
    plt.savefig(f"figures/Halton-{N}-samples.png")
    plt.title("Halton")
    plt.show()


        

     
