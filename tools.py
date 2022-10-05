import matplotlib.pyplot as plt
import numpy as np


def euclidian_dist(a, b):
    return np.linalg.norm(a-b)


def plot_cat_and_par(X1, X2, par, cat, delta, path_to_img=None, show_img=False):

    A, B, C = par
    N = X2 - X1 # Assuming the exps are generated in experiments.py with P1_x < P2_x < P3_x        
    Xs, Ys_par = [], []
    for i in range(int(N/10**-2)):
        x = X1 + i*delta            
        par_y = A*x**2+B*x+C

        Xs.append(x)
        Ys_par.append(par_y)

    cat.plot2D()
    plt.plot(Xs, Ys_par, "-r")
    if path_to_img:
        plt.savefig(path_to_img)
    if show_img:
        plt.show()

    plt.close()


def par_cat_comparison(X1, X2, par, cat, delta):
    A, B, C = par
    N = X2 - X1 # Assuming the exps are generated in experiments.py with P1_x < P2_x < P3_x        
    Xs, Ys_par = [], []
    for i in range(int(N/delta)):
        x = X1 + i*delta            
        par_y = A*x**2+B*x+C

        Xs.append(x)
        Ys_par.append(par_y)

    dd, hh = [], []
    ss = np.linspace(0., np.sum(cat.L), int(N/delta))
    for s in ss:
        xyz = cat.s2xyz(s)
        dd.append(np.sqrt(xyz[0]**2+xyz[1]**2))
        hh.append(xyz[2])

    diffs = [abs(z1-z2) for z1, z2 in zip(Ys_par, hh)]
    total = sum(diffs)

    return {"total": total, "max": max(diffs), "mean": total/len(diffs)}