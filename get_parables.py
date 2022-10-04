import matplotlib.pyplot as plt
import numpy as np

from tools import *

def calc_parabola_vertex(x1, y1, x2, y2, x3, y3, plotting = False):
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


def approx_parabola_length(P1, P2, P3, A, B, C, delta=10**-4):
    
    A, B, C = coef
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


if __name__ == "__main__":

    P1=[-20, 0]
    P2=[-1,-1]
    P3=[3,4]

    coef, eq = calc_parabola_vertex(P1[0], P1[1], P2[0], P2[1], P3[0], P3[1], plotting=True)
    # print(eq)

    A, B, C = coef
    L = approx_parabola_length(P1, P2, P3, A, B, C)

    print(L)
