import bisect
from cmath import sinh
import matplotlib.pyplot as plt
import numpy as np


def euclidian_dist(a, b):
    return np.linalg.norm(a-b)


def plot_cat_and_par(X1, X2, par, cat, delta, path_to_img=None, show_img=False):

    A, B, C = par
    N = X2 - X1 # Assuming the exps are generated in experiments.py with P1_x < P2_x < P3_x        
    Xs, Ys_par = [], []
    for i in range(int(N/delta)):
        x = X1 + i*delta            
        par_y = A*x**2+B*x+C

        Xs.append(x)
        Ys_par.append(par_y)

    cat.plot2D()
    plt.plot(Xs, Ys_par, label="parable")
    plt.legend()
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

    # diffs = [abs(z1-z2) for z1, z2 in zip(Ys_par, hh)]
    # total = sum(diffs)
    # max_diff = max(diffs)
    # i_max = diffs.index(max_diff)
    # x_max = X1 + i_max*delta
    # j_max = bisect.bisect_left(dd, x_max)

    diffs = []
    Xs_cat = []
    Zs_cat = []
    for x, y in zip(Xs, Ys_par):
        j = bisect.bisect_left(dd, x)
        diffs.append(abs(hh[j]-y))
        Xs_cat.append(dd[j])
        Zs_cat.append(hh[j])

    total = sum(diffs)
    max_diff = max(diffs)
    i_max = diffs.index(max_diff)
        
    return {
        "total": total, 
        "max": max_diff, 
        "mean": total/len(diffs), 
        "x_max": Xs[i_max], 
        "max_x_cat": Xs_cat[i_max],
        "max_z_cat": Zs_cat[i_max]
    }


def generic_functions_comparison(f1, f2, range_, delta, plotting=None):
    
    a, b = range_
    Xs = []
    Ys1, Ys2 = [], []
    diffs = []
    for i in range(int((b-a)/delta)):
        x = a+delta*i
        Xs.append(x)
        Ys1.append(f1(x))
        Ys2.append(f2(x))
        
        diffs.append(abs(f1(x)-f2(x)))
    
    if plotting:
        plt.plot(Xs, Ys1, label="f1")
        plt.plot(Xs, Ys2, label="f2")
        plt.legend()
        plt.show()

    total = sum(diffs)
    return {"total": total, "max": max(diffs), "mean": total/len(diffs)}

def area_under_curve(curve_waypoints):
    area = 0
    for i in range(len(curve_waypoints)-1):
        p1 = curve_waypoints[i]
        p2 = curve_waypoints[i+1]

        h = p2[0] - p1[0]
        area += h*(p1[1]+p2[1])/2

    return area


def get_par_eq_from_area(A, B, area):
    # yA = pxA^2 + qxA + r
    # yB = pxB^2 + qxB + r
    # area = int(A, B, px^2+qx+r)
    # F = px^3/3 + qx^2/2 + rx
    # area = p(xA^3-xB^3)/3 + q(xA^2-xB^2)/2 + r(xA-xB)

    M = np.array([
        [A[0]**2, A[0], 1],
        [B[0]**2, B[0], 1],
        [(A[0]**3-B[0]**3)/3, (A[0]**2-B[0]**2)/2, A[0]-B[0]]
    ])
    b = np.array([A[1], B[1], area])

    return np.linalg.solve(M, b)

    
