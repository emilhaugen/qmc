from inspect import signature
import numpy as np 
import pandas as pd
import scipy.special as special
import scipy.integrate as integrate 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from haltonscale import halton_nD


def halton_integrator(f, low, high, N=100, seed=110):
    """ 
    Integrate f on interval (low, high) or square (low, high)^2 according to dimension d.  
    """
    np.random.seed(seed)
    d = len(signature(f).parameters) # read dimension from integrand
    
    true = integrate.nquad(f, ranges = [[low, high] for i in range(d)])[0]
    
    Halton_samples = halton_nD(N, low=low, high=high, d=d).values
    Random_samples = np.random.uniform(low, high, (N, d))
    
    integrals = pd.DataFrame(np.zeros(shape=(N, 2)), columns = ["Halton", "Random"])
    integrals.Halton[0] = f(*Halton_samples[0,:]) # one sample estimate
    integrals.Random[0] = f(*Random_samples[0,:])
    
    for n in range(1, N): # calculate unscaled estimates for 1 to N samples 
        integrals.Halton[n] = f(*Halton_samples[n-1,:]) + integrals.Halton[n-1]
        integrals.Random[n] = f(*Random_samples[n-1,:]) + integrals.Random[n-1]    
    for n in range(N): # scale estimates afterwards
        integrals.Halton[n] = (high-low)**d*integrals.Halton[n] / (n+1)  
        integrals.Random[n] = (high-low)**d*integrals.Random[n] / (n+1)  
    
    return {"integrals":integrals, "true":true, "d":d}

def halton_plot(f, f_label="$f(x)$", low=0, high=1, N=100, filename="tmp"):
    
    result = halton_integrator(f, low=low, high=high, N=N)
    true_val = result["true"]
    d = result["d"]
    
    ind = np.linspace(1, N, N)
    halton_ints = result["integrals"]["Halton"]
    random_ints = result["integrals"]["Random"]

    # plot integrals
    fig, ax = plt.subplots()
    ax.plot(ind, true_val*np.ones(N), 'r', label="True value")
    ax.scatter(ind, halton_ints, s=1, c="b", label="Halton sampling")
    ax.scatter(ind, random_ints, s=1, c="orange", label="Random sampling")
    ax.set_xlim([int(np.log(N)*0.1), N]) # drop first estimates 
    #ax.set_ylim([11.5, 14])
    ax.set_xlabel("Number of samples")
    ax.legend(loc="upper right", markerscale=6)
    fig.savefig(f"figures/{filename}-integrals.eps")
    plt.show()

    if d == 1:
        fig, ax = plt.subplots()
        t = np.linspace(low, high, 1000)
        ax.plot(t, f(t), label=f_label)
        #ax.legend(loc="upper right")
        fig.savefig(f"{filename}-graph.eps")
        plt.show()
    if d == 2:
        fig = plt.figure(figsize=(10,10), constrained_layout=True)
        ax = plt.axes(projection="3d")
        x = np.linspace(low, high, 1000)
        X, Y = np.meshgrid(x, x)
        z = f(X, Y)        
        ax.contour3D(X, Y, z, 200)
        ax.view_init(18, 166)
        fig.savefig(f"figures/{filename}-graph.eps", bbox_inches='tight')
        plt.show()

if __name__=="__main__":
    
    n, alpha = 2, 2
    laguerre = special.genlaguerre(n, alpha)
    
    N = 10000

    #true function and integral
    f = lambda x : x**alpha * laguerre(x)**2 * np.exp(-x)
    x_str = f"x^{alpha}" if alpha > 1 else "x"
    f_label = f"$f(x) = {x_str} \cdot L^{alpha}_{n}(x)^2 \cdot \exp(-x)$"    
    plot_name = "Laguerre-halton"
    halton_plot(f,f_label, low=0, high=20, N=N, filename=plot_name)

    f = lambda x,y : np.exp(-x**2 - y**2) + (x+y)**2
    low, high = -1, 1    
    plot_name = "2D"

    halton_plot(f, low=low, high=high, N=N, filename=plot_name)


    






