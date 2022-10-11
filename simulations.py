from cmath import sinh
import json
import matplotlib.pyplot as plt
import numpy as np

from get_catenaries import get_cat_btwn_2points
from get_parables import get_par_from_3points, approx_parable_length, get_parable_vertex_form
from tools import *


def simulate(path_to_data="data/experiments.json", delta=10**-4, plotting=False, show_img=False):
    
    with open(path_to_data, "r") as f:
        exps = json.loads(f.read())

    slm_name = "Same Length Method"
    vfcm_name = "Vertex Form Coefficients Method"
    results = {
        slm_name: {"mean":[], "max":[], "total":[]},
        vfcm_name: {"mean":[], "max":[], "total":[]}
    }
    for i, exp in enumerate(exps):
        print(f"**Experiment {i}")

        P1 = np.array(exp["A"])        
        P2 = np.array(exp["B"])        
        P3 = np.array(exp["C"]) 

        coefs, eq = get_par_from_3points(P1[0], P1[1], P2[0], P2[1], P3[0], P3[1])       

        A, B, C = coefs

        # Length comparison
        L = approx_parable_length(P1, P2, P3, A, B, C)
        cat = get_cat_btwn_2points(P1, P3, L, path_to_fig=None)
        
        if plotting:
            plot_cat_and_par(P1[0], P3[0], coefs, cat, delta, path_to_fig=f"figs/exps/{slm_name}_{i}.jpg")

        diffs = par_cat_comparison(P1[0], P3[0], coefs, cat, delta)
        for k, v in diffs.items():
            results[slm_name][k].append(v)

        # Vertex form Eq comparison
        vertex_coef, eq_vertex = get_parable_vertex_form(A, B, C)
        A, H, K = vertex_coef

        par = lambda x: A*(x-H)**2+K
        cat2 = lambda x: (1/(2*A))*(sinh(A*(x-H))**2 + K) 

        diffs = generic_functions_comparison(par, cat2, [P1[0], P3[0]], delta)
        for k, v in diffs.items():
            results[vfcm_name][k].append(v)


    for method in results:
        for k, v in results[method].items():
            results[method][k] = sum(v)/len(v)
        
    print(results)
    with open("data/results.json", "w+") as f: 
        f.write(json.dumps(results))

    return results

if __name__ == "__main__":

    simulate(show_img=False, delta=10**-2)

