import matplotlib.pyplot as plt
import numpy as np

from tools import *


def get_par_from_3points(x1, y1, x2, y2, x3, y3, plotting = False):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1-x2) * (x1-x3) * (x2-x3);
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

    if plotting:
        axis = [x1, x2, x3]
        Xmin = min(axis)
        Xmax = max(axis)

        Xs = [Xmin+i/100*(Xmax-Xmin) for i in range(100)]
        Ys = [A*x**2+B*x+C for x in Xs]

        plt.plot(Xs, Ys, "-b")
        plt.plot(axis, [y1, y2, y3], "or")
        plt.show()

    return [A, B, C], f"({A})x^2 + ({B})x + ({C})"


def approx_parable_length(P1, P2, P3, A, B, C, delta=10**-4):
    
    axis = [P1[0], P2[0], P3[0]]
    Xmin = min(axis) 
    Xmax = max(axis)

    N = Xmax - Xmin
    L = 0
    P_prev = np.array([])
    for i in range(int(N/delta)):
        x = Xmin + i*delta
        y = A*x**2+B*x+C
        
        if len(P_prev) != 0: 
            L += euclidian_dist(np.array([x, y]), P_prev)
        
        P_prev = np.array([x, y])

    return L    


def parabola_sag(P1, P3, A, B, C):
    x1, y1 = P1
    x3, y3 = P3
    
    # vertex of parabola
    xv = -B / (2*A)
    yv = A*xv**2 + B*xv + C
    
    # chord line evaluated at xv
    yc = y1 + (y3 - y1)/(x3 - x1) * (xv - x1)
    
    # sag is the vertical distance
    sag = yc - yv
    return sag

def parabola_max_sag(P1, P3, A, B, C, npoints=1000):
    x1, y1 = P1
    x3, y3 = P3
    
    # parametric chord line
    m = (y3 - y1) / (x3 - x1) if x3 != x1 else None
    
    xs = np.linspace(min(x1, x3), max(x1, x3), npoints)
    ys_par = A*xs**2 + B*xs + C
    
    if m is not None:
        ys_chord = y1 + m * (xs - x1)
        dists = np.abs(ys_chord - ys_par)   # vertical distance
    else:
        # vertical chord case: distance is horizontal
        dists = np.abs(xs - x1)
    
    idx = np.argmax(dists)
    return dists[idx]


def get_parable_vertex_form(A, B, C):
    
    h = -B/(2*A)
    k = A*h**2 + B*h + C

    return [A, h, k], f"y={A}*(x-({h}))^2+{k}"



def get_parable_vertex_from_origin(P1, P2, a0):
    
    # Assuming x3 > x1 and y3 > y1
    x1, h1 = P1
    x2, h2 = P2
    S = x2 - x1
    
    xmin = S/2 - (h2-h1)/(2*a0*S)
    ymin = h1 - a0*(xmin)**2
    
    return [a0, xmin, ymin], f"y={A}*(x-({xmin}))^2+{ymin}"



if __name__ == "__main__":

    P1=[-10, 1]
    P2=[3,-6]
    P3=[6,6]

    coef, eq = get_par_from_3points(P1[0], P1[1], P2[0], P2[1], P3[0], P3[1], plotting=False)
    # print(eq)

    A, B, C = coef
    # L = approx_parable_length(P1, P2, P3, A, B, C)

    # print(L)

    # coef2, s = get_parable_vertex_form(A, B, C)
    coef2, s = get_parable_vertex_from_origin(P1, P3, A)
    # print(s)
    
    a0, h,k = coef2

    Xs = [-50+i for i in range(101)]
    Ys1 = [A*x**2+B*x+C for x in Xs]
    Ys2 = [A*(x-h)**2+k for x in Xs]

    plt.plot(Xs,Ys1,"b")
    plt.plot(Xs,Ys2,"--r")
    plt.show()

