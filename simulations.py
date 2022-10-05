import json
import matplotlib.pyplot as plt
import numpy as np

from get_catenaries import get_cat_btwn_2points
from get_parables import get_par_from_3points, approx_parabola_length
from tools import *


def simulate(path_to_data="data/experiments.json", delta=10**-4, plotting=False, show_img=False):
    
    with open(path_to_data, "r") as f:
        exps = json.loads(f.read())

    for i, exp in enumerate(exps):
        P1 = np.array(exp["A"])        
        P2 = np.array(exp["B"])        
        P3 = np.array(exp["C"]) 

        coefs, eq = get_par_from_3points(P1[0], P1[1], P2[0], P2[1], P3[0], P3[1])       

        A, B, C = coefs
        L = approx_parabola_length(P1, P2, P3, A, B, C)
        print("longitude", L)
        cat = get_cat_btwn_2points(P1, P3, L, path_to_fig=None)
        
        if plotting:
            plot_cat_and_par(P1[0], P3[0], coefs, cat, delta)

        N = P3[0] - P1[0] # Assuming the exps are generated in experiments.py with P1_x < P2_x < P3_x

        par_cat_comparison(P1[0], P3[0], coefs, cat, delta)



if __name__ == "__main__":

    simulate(show_img=False)

