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


def touch_the_ground(f, interval, delta=10**-3):

    X1, X2 = interval
    for i in range(int((X2-X1)/delta)):
        if f(X1+i*delta) <= 0:
            return True

    return False

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
        [(B[0]**3-A[0]**3)/3, (B[0]**2-A[0]**2)/2, B[0]-A[0]]
    ])
    b = np.array([A[1], B[1], area])

    return np.linalg.solve(M, b)


def get_par_eq_from_area_example(A, B, area):
    # yA = pxA^2 + qxA + r
    # yB = pxB^2 + qxB + r
    # area = int(A, B, px^2+qx+r)
    # F = px^3/3 + qx^2/2 + rx
    # area = p(xA^3-xB^3)/3 + q(xA^2-xB^2)/2 + r(xA-xB)

    M = np.array([
        [A[0]**2, A[0], 1],
        [B[0]**2, B[0], 1],
        [(B[0]**3-A[0]**3)/3, (B[0]**2-A[0]**2)/2, B[0]-A[0]]
    ])
    b = np.array([A[1], B[1], area])

    print('M', M)
    print('b', b)
    print('det', np.linalg.det(M))
    print('lstsq', np.linalg.lstsq(M, b)[0])
    print('solve', np.linalg.solve(M, b))

    p, q, r = np.linalg.solve(M, b)
    p2, q2, r2 = 1,0,0
    p3, q3, r3 = np.linalg.lstsq(M, b)[0]

    Xs, Ys_par, Ys_par2, Ys_par3 = [], [], [], []
    for i in range(int(4/10**-5)):
        x = A[0] + i*10**-5            
        par_y = p*x**2+q*x+r
        par_y2 = p2*x**2+q2*x+r2
        par_y3 = p3*x**2+q3*x+r3

        Xs.append(x)
        Ys_par.append(par_y)
        Ys_par2.append(par_y2)
        Ys_par3.append(par_y3)

    plt.plot(Xs, Ys_par, '--')
    plt.plot(Xs, Ys_par2, '--')
    plt.plot(Xs, Ys_par3, label="lstsq")
    plt.show()

    return np.linalg.solve(M, b)

    
if __name__ == "__main__":

    coef = get_par_eq_from_area((0,0), (3,9), 10)
    print(coef)