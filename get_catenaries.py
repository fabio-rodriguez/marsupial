import numpy as np

from pycatenary import cable
from tools import *


def example():

    # define properties of cable
    length = 6.98  # length of line
    # w = 1.036  # submerged weight (elastic catenary)
    w = 0  # submerged weight (rigid Catenary)
    # EA = 560e3  # axial stiffness (elastic catenary)
    EA = None  # axial stiffness (rigid Catenary)
    # floor = True  # if True, contact is possible at the level of the anchor
    floor = False  # if True, contact is possible at the level of the anchor
    anchor = [0., 0., 0.]
    fairlead = [5.2, 1., 2.65]

    # create cable instance
    l1 = cable.MooringLine(L=length,
                        w=w,
                        EA=EA,
                        anchor=anchor,
                        fairlead=fairlead,
                        floor=floor)

    # compute calculations
    l1.computeSolution()

    # change fairlead position
    l1.setFairleadCoords([5.4, 1., 2.65])

    # recompute solution
    l1.computeSolution()

    # get tension along line (between 0. and total line length)
    s = 5.
    T = l1.getTension(s)

    # get xyz coordinates along line
    xyz = l1.s2xyz(s)

    # plot cable cable.MooringLine instance l1
    l1.plot2D(path_to_fig="figs/2D_example.jpg")
    
    # plot cable cable.MooringLine instance l1
    l1.plot3D(path_to_fig="figs/3D_example.jpg")


def get_cat_btwn_2points(point1, point2, length, path_to_fig = "figs/plotting.jpg"):

    w = 0  # submerged weight (rigid Catenary)
    EA = None  # axial stiffness (rigid Catenary)
    floor = False  # if True, contact is possible at the level of the anchor
    # anchor = [0., 0., 0.]
    # diff = point2-point1
    # fairlead = [diff[0], 0., diff[1]]
    anchor = [point1[0], 0., point1[1]]
    fairlead = [point2[0], 0., point2[1]]

    # create cable instance
    l1 = cable.MooringLine(L=length,
                        w=w,
                        EA=EA,
                        anchor=anchor,
                        fairlead=fairlead,
                        floor=floor)

    # compute calculations
    l1.computeSolution()
    
    # plot cable cable.MooringLine instance l1
    if path_to_fig:
        l1.plot2D(path_to_fig=path_to_fig)
    
    return l1


def approx_optimal_cat(P1, P3, par, max_cat_len, delta, path_to_fig):

    A, B, C = par
    min_cat_len = euclidian_dist(P1, P3)
    while True:
        mid_len = (max_cat_len + min_cat_len)/2
        mid_cat = get_cat_btwn_2points(P1, P3, mid_len, path_to_fig=None)
        
        if max_cat_len - min_cat_len < delta:
            cat_result=mid_cat
            break 

        # greater distance axis
        # diffs = par_cat_comparison(P1[0], P3[0], par, mid_cat, delta)
        diffs = par_cat_comparison(P1[0], P3[0], par, mid_cat, delta)
        x_max = diffs["x_max"] 
        y_par = A*x_max**2+B*x_max+C 
        
        z_cat = diffs["max_z_cat"] 
        
        if  y_par > z_cat:
            max_cat_len = mid_len
        else:
            min_cat_len = mid_len

    if path_to_fig:
        cat_result.plot2D(path_to_fig=path_to_fig)

    return cat_result  




if __name__ == "__main__":

    # example()

    A = np.array([50,50])
    B = np.array([100, 70])
    L = 250
    # longitude 12.921033822098877
    # [3.64 7.76] [6.57 8.45]
    # # Example 1
    # A = np.array([0.22, 7.91])
    # B = np.array([5.87, 8.37])
    # L = 9.98
    # # Example 2 
    A = np.array([2.49, 7.76])
    B = np.array([9, 7])
    L = 15
    
    # get_cat_btwn_2points(A,B,L,None)
    get_cat_btwn_2points(A,B,L)
