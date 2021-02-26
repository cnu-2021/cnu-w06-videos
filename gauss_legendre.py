# Numerical investigation of properties of quadrature rules
# Example: Gauss-Legendre quadrature

import numpy as np
import matplotlib.pyplot as plt

compute_dop = False
convergence = True

# Calculate nodes and weights for N nodes
N = 3
xk, wk = np.polynomial.legendre.leggauss(N)


if compute_dop:
    # Define convenience functions for degree n polynomial and integral
    def p(x, n):
        return x**n

    def p_int(x, n):
        return x**(n+1) / (n+1)


    # First: degree of precision.
    n = 0
    tol = 1e-13

    while n <= 100:
        # Test x**n
        I_exact = p_int(1, n) - p_int(-1, n)
        
        # I_approx = 0
        # for i in range(N):
        #     I_approx += wk[i] * p(xk[i], n)
        
        I_approx = np.sum(wk * p(xk, n))
        
        # I_approx = wk @ p(xk, n)
        # I_approx = np.dot(wk, p(xk, n))
        
        err = abs(I_exact - I_approx)
        if err > tol:
            print(f'The degree of precision for {N} nodes is {n-1}.')
            break
        
        n += 1
    
if convergence:
    # Rate of convergence (composite)

    # Number of intervals
    M_vals = np.logspace(1, 7, 7, base=2, dtype=int)
    err = []
    h_vals = 2 / M_vals
    
    # Define a random polynomial
    p_coeffs = 2 * np.random.random(2*N + 2) - 1
        
    # Calculate exact integral
    p_coeffs_int = np.polyint(p_coeffs)
    I_exact = np.polyval(p_coeffs_int, 1) - np.polyval(p_coeffs_int, -1)
    
    for M in M_vals:

        # Subinterval boundaries
        bounds = np.linspace(-1, 1, M+1)

        # Calculate approximation
        I_approx = 0
        for i in range(M):
            yk = (bounds[i+1] - bounds[i]) / 2 * (xk + 1) + bounds[i]
            I_approx += (bounds[i+1] - bounds[i]) / 2 * np.sum(wk * np.polyval(p_coeffs, yk))
        
        # Compute the error
        err.append(abs(I_exact - I_approx))


    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # First plot: plot err vs h, set axis scale to log
    ax[0].plot(h_vals, err, 'rx')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set(title='Log axes', xlabel='h', ylabel='Error')
    
    # Second plot: plot log(err) vs log(h), leave axis scale linear
    ax[1].plot(np.log(h_vals), np.log(err), 'b+')
    ax[1].set(title='Log values', xlabel='log(h)', ylabel='log(err)')
    
    plt.show()
    
    line_coeffs = np.polyfit(np.log(h_vals), np.log(err), 1)
    print(line_coeffs[0])
