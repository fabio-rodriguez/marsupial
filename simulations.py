from cmath import asinh, sinh
import json
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
import time

from get_catenaries import get_cat_btwn_2points, approx_optimal_cat
from get_paraboles import get_par_from_3points, approx_parable_length, get_parable_vertex_form, get_parable_vertex_from_origin
from tools import *


def plot_parabolas(path_to_data="data/experiments.json", delta=10**-3):

    with open(path_to_data, "r") as f:
        exps = json.loads(f.read())

    for i, exp in enumerate(exps):
        print(f"**Experiment {i}")

        P1 = np.array(exp["A"])        
        P2 = np.array(exp["B"])        
        P3 = np.array(exp["C"]) 

        coefs, eq = get_par_from_3points(P1[0], P1[1], P2[0], P2[1], P3[0], P3[1])       
        A, B, C = coefs
        
        N = P3[0] - P1[0] # Assuming the exps are generated in experiments.py with P1_x < P2_x < P3_x        
        Xs, Ys_par = [], []
        for i in range(int(N/delta)):
            x = P1[0] + i*delta            
            par_y = A*x**2+B*x+C

            Xs.append(x)
            Ys_par.append(par_y)

        plt.plot(Xs, Ys_par, label="parable")
        plt.legend()
        plt.show()
        plt.close()


def simulate(path_to_data="data/experiments.json", max_cat_len = 60, delta=10**-4, plotting=False, show_img=False):
    
    with open(path_to_data, "r") as f:
        exps = json.loads(f.read())

    slm_name = "Same Length Method"
    vfcm_name = "Vertex Form Coefficients Method"
    optm_name = "Optimal Catenary Method"
    results = {
        slm_name: {"mean":[], "max":[], "total":[], "time": []},
        vfcm_name: {"mean":[], "max":[], "total":[], "time": []},
        optm_name: {"mean":[], "max":[], "total":[], "time": []}
    }

    for i, exp in enumerate(exps[99:]):
        print(f"**Experiment {i}")

        P1 = np.array(exp["A"])        
        P2 = np.array(exp["B"])        
        P3 = np.array(exp["C"]) 

        coefs, eq = get_par_from_3points(P1[0], P1[1], P2[0], P2[1], P3[0], P3[1])       
        A, B, C = coefs

        # Length comparison
        L = approx_parable_length(P1, P2, P3, A, B, C)
        t = time.time()
        cat = get_cat_btwn_2points(P1, P3, L, path_to_fig=None)
        results[slm_name]["time"].append(time.time()-t)
        
        # if plotting:
        #     plot_cat_and_par(P1[0], P3[0], coefs, cat, delta, path_to_img=f"figs/exps/{slm_name}_{i}.jpg")


        diffs = par_cat_comparison(P1[0], P3[0], coefs, cat, delta)
        for k, v in diffs.items():
            if k in results[slm_name]:
                results[slm_name][k].append(v) 

        # Vertex form Eq comparison
        vertex_coef, eq_vertex = get_parable_vertex_form(A, B, C)
        A, H, K = vertex_coef

        par = lambda x: A*(x-H)**2+K
        t = time.time()        
        S = P3[0]-P1[0]
        c = 1/(2*A)
        xmin = S/2 - c*asinh((P3[1]-P1[1])/(2*c*sinh(S/(2*c))))
        ymin = P1[1] - 2*c*sinh(xmin/(2*c))**2
        cat2 = lambda x: 2*c*sinh((x-xmin.real)/(2*c))**2+ymin.real
        results[vfcm_name]["time"].append(time.time()-t)

        # if plotting:
        #     Xs = [i for i in range(int(P3[0]+2))]
        #     Ys1 = [A*x**2+B*x+C for x in Xs]
        #     Ys3 = [cat2(x) for x in Xs]

        #     plt.plot(Xs,Ys1,label=f"{round(A, 3)}x^2+{round(B, 3)}x+{round(C, 3)}")
        #     plt.plot(Xs,Ys3,label=f"x_min={round(xmin.real,3)}, y_min={round(ymin.real,3)}, c={round(c,3)}")
        #     plt.legend()
        #     plt.savefig(f"figs/vertex_coef_method/{i}.jpg")
        #     plt.close()

        # Approx optimal catenary
        t = time.time() 
        opt_cat = approx_optimal_cat(P1, P3, coefs, max_cat_len, delta=delta, path_to_fig=None)
        results[optm_name]["time"].append(time.time()-t)
        
        # if plotting:
        #     plot_cat_and_par(P1[0], P3[0], coefs, opt_cat, delta, path_to_img=f"figs/opt_method/{optm_name}_{i}.jpg")

        diffs = par_cat_comparison(P1[0], P3[0], coefs, opt_cat, delta)
        for k, v in diffs.items():
            if k in results[optm_name]:
                results[optm_name][k].append(v)

        if plotting:
            # Cat 1
            N = P3[0] - P1[0] # Assuming the exps are generated in experiments.py with P1_x < P2_x < P3_x        
            Xs, Ys_par = [], []
            for i in range(int(N/delta)):
                x = P1[0] + i*delta            
                par_y = A*x**2+B*x+C

                Xs.append(x)
                Ys_par.append(par_y)

            plt.plot(Xs, Ys_par, label="Parabola", color="blue")


            # Cat 2
            Xs = [i for i in range(int(P3[0]))]
            Ys3 = [cat2(x) for x in Xs]
            plt.plot(Xs+[P3[0]],Ys3+[cat2(P3[0])],label=f"ByPSeries", color="green")
  
            ##
            xyzs = []
            dd = []
            hh = []
            npoints=100
            ds = np.sum(cat.L)/(npoints-1)
            ss = np.linspace(0., np.sum(cat.L), npoints)
            for s in ss:
                xyz = cat.s2xyz(s)
                xyzs.append(xyz)
                dd.append(np.sqrt(xyz[0]**2+xyz[1]**2))
                hh.append(xyz[2])
            plt.plot(dd, hh, label="ByLength", color="orange")
              
            ##
            xyzs = []
            dd = []
            hh = []
            npoints=100
            ds = np.sum(opt_cat.L)/(npoints-1)
            ss = np.linspace(0., np.sum(opt_cat.L), npoints)
            for s in ss:
                xyz = opt_cat.s2xyz(s)
                xyzs.append(xyz)
                dd.append(np.sqrt(xyz[0]**2+xyz[1]**2))
                hh.append(xyz[2])
            plt.plot(dd, hh, label="ByFitting", color="red")
        
            # opt_cat.plot2D(label=)
            
            plt.grid()
            plt.legend()
            plt.show()
            plt.close()


    clean_results = {}
    for method in [slm_name, vfcm_name, optm_name]:
        clean_results[method] = {"total_max": max(results[method]["max"])}
        for k, v in results[method].items():
            clean_results[method][f"{k}_std"] = np.std(v)
            clean_results[method][k] = np.mean(v)

    print(clean_results)
    with open("data/results.json", "w+") as f: 
        f.write(json.dumps(results))

    return results


def area_comparison(path_to_data="data/experiments.json", delta=10**-4):
    
    with open(path_to_data, "r") as f:
        exps = json.loads(f.read())

    results = {}
    print(f'Total Exp: {len(exps)}')
    for i, exp in enumerate(exps):
        print(f'*** Exp {i}')

        P1 = np.array(exp["A"])        
        P2 = np.array(exp["B"])        
        P3 = np.array(exp["C"]) 

        coefs, eq = get_par_from_3points(P1[0], P1[1], P2[0], P2[1], P3[0], P3[1])       
        A, B, C = coefs

        L = approx_parable_length(P1, P2, P3, A, B, C)
        cat = get_cat_btwn_2points(P1, P3, L, path_to_fig=None)

        xyzs = []
        dd = []
        hh = []
        ss = np.linspace(0., np.sum(cat.L), int(abs((P2[0]-P1[0]))/delta))
        for s in ss:
            xyz = cat.s2xyz(s)
            xyzs.append(xyz)
            dd.append(np.sqrt(xyz[0]**2+xyz[1]**2))
            hh.append(xyz[2])
    
        area = area_under_curve(list(zip(dd, hh)))
        
        # print(area)
        # plt.plot([P1[0], P3[0]], [P1[1], P3[1]], 'or')
        # plt.plot(dd, hh)
        # plt.show()

        coefs = get_par_eq_from_area(P1, P3, area)
        diffs = par_cat_comparison(P1[0], P3[0], coefs, cat, delta)
        # print(diffs)
        # print(coefs)
        # plot_cat_and_par(P1[0], P3[0], coefs, cat, delta, show_img=True)#path_to_img=f'{i}.jpg')
        
        p, q, r = coefs
        # par = lambda x: p*x**2+q*x+r
        # diffs = generic_functions_comparison(par, cat, [P1[0], P3[0]], delta)
        diffs = par_cat_comparison(P1[0], P3[0], coefs, cat, delta)
        
        for k, v in diffs.items():
            try:
                results[k].append(v)
            except:
                results[k] = [v]

    clean_results = {}
    for k, v in results.items():
        clean_results[f"{k}_std"] = np.std(v)
        clean_results[k] = np.mean(v)

    print(clean_results)





if __name__ == "__main__":

    simulate(show_img=False, delta=10**-2)

    # plot_parabolas()

    # area_comparison(delta=10**-2)
