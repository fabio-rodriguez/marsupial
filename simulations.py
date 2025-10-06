from cmath import asinh, sinh
import json
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
import time

from get_catenaries import get_cat_btwn_2points, approx_optimal_cat
from get_paraboles import get_par_from_3points, approx_parable_length, get_parable_vertex_form, get_parable_vertex_from_origin, parabola_sag, parabola_max_sag
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

    L_length, L_coeff, L_opt = {}, {}, {}
    for i, exp in enumerate(exps):
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
        

        # Compute sag vs error
        # si = abs(parabola_sag(P1, P3, A, B, C))
        si = abs(parabola_max_sag(P1, P3, A, B, C))
        L_length[si] = []
        L_coeff[si] = []
        L_opt[si] = []


        # if plotting:
        #     plot_cat_and_par(P1[0], P3[0], coefs, cat, delta, path_to_img=f"figs/exps/{slm_name}_{i}.jpg")


        diffs = par_cat_comparison(P1[0], P3[0], coefs, cat, delta)
        for k, v in diffs.items():
            if k in results[slm_name]:
                results[slm_name][k].append(v) 
        
        L_length[si].append(diffs["mean"])

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

        diffs = par_cat_error(P1, P3, coefs, cat2, delta)
        for k, v in diffs.items():
            if k in results[vfcm_name]:
                results[vfcm_name][k].append(v)

        L_coeff[si].append(diffs["mean"])

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

        L_opt[si].append(diffs["mean"])
        

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

    print()
    print("******************")
    print("******************")
    print()

    Lclean_length = []
    for k, v in L_length.items():
        Lclean_length.append((k, sum(v)/len(v)))

    Lclean_coeff = []
    for k, v in L_coeff.items():
        Lclean_coeff.append((k, sum(v)/len(v)))

    Lclean_opt = []
    for k, v in L_opt.items():
        Lclean_opt.append((k, sum(v)/len(v)))

    # Sort by sag value (k)
    Lclean_length.sort(key=lambda x: x[0])
    Lclean_coeff.sort(key=lambda x: x[0])
    Lclean_opt.sort(key=lambda x: x[0])

    # ---- Plotting ----
    plt.figure(figsize=(8,5))
    plt.plot([x for x,y in Lclean_length], [y for x,y in Lclean_length], label="Length Method", marker="o")
    plt.plot([x for x,y in Lclean_coeff], [y for x,y in Lclean_coeff], label="Vertex Coeff Method", marker="s")
    plt.plot([x for x,y in Lclean_opt], [y for x,y in Lclean_opt], label="Optimal Catenary", marker="^")

    plt.xlabel("Sag")
    plt.ylabel("Mpar_cat_comparisonean Error")
    plt.title("Mean Error vs Sag")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/mean_error_vs_sag.png")
    plt.show()
    plt.close()

    # ---- Save to JSON ----
    output_data = {
        "length": Lclean_length,
        "coeff": Lclean_coeff,
        "opt": Lclean_opt
    }




    bin_centers, mean_length = bin_sag_data(Lclean_length)
    bin_centers, mean_coeff  = bin_sag_data(Lclean_coeff)
    bin_centers, mean_opt    = bin_sag_data(Lclean_opt)

    # Smooth each series
    bin_centers, mean_length_smooth = fill_missing_bins(bin_centers, mean_length)
    _, mean_coeff_smooth = fill_missing_bins(bin_centers, mean_coeff)
    _, mean_opt_smooth   = fill_missing_bins(bin_centers, mean_opt)

    plt.figure(figsize=(8,5))
    plt.plot(bin_centers, mean_length_smooth, label="Length Method", marker="o")
    plt.plot(bin_centers, mean_coeff_smooth, label="Vertex Coeff Method", marker="s")
    plt.plot(bin_centers, mean_opt_smooth, label="Optimal Catenary", marker="^")

    plt.xlabel("Sag (binned)")
    plt.ylabel("Mean Error")
    plt.title("Mean Error vs Sag (Binned & Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ---- Compute binned and smoothed curves ----
    bin_centers, mean_length_smooth = fill_missing_bins(bin_centers, mean_length)
    bin_centers, mean_coeff_smooth = fill_missing_bins(bin_centers, mean_coeff)
    bin_centers, mean_opt_smooth   = fill_missing_bins(bin_centers, mean_opt)

    # ---- Normalize ----
    all_values = np.concatenate([mean_length_smooth, mean_coeff_smooth, mean_opt_smooth])
    global_max = np.nanmax(all_values)  # safeguard in case of NaNs

    mean_length_norm = mean_length_smooth / global_max
    mean_coeff_norm  = mean_coeff_smooth / global_max
    mean_opt_norm    = mean_opt_smooth   / global_max

    # ---- Plotting ----
    plt.figure(figsize=(8,5))
    plt.plot(bin_centers, mean_length_norm, label="Length Method", marker="o")
    plt.plot(bin_centers, mean_coeff_norm, label="Vertex Coeff Method", marker="s")
    plt.plot(bin_centers, mean_opt_norm, label="Optimal Catenary", marker="^")

    plt.xlabel("Sag")
    plt.ylabel("Normalized Mean Error (0–1)")
    plt.title("Normalized Mean Error vs Sag")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/normalized_mean_error_vs_sag.png")
    plt.show()

    with open("results/mean_error_vs_sag.json", "w") as f:
        json.dump(output_data, f, indent=4)

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



def bin_sag_data(Lclean, bin_size=1, max_sag=12):
    """
    Group sag-error data into bins and compute the mean error for each bin.
    
    Parameters
    ----------
    Lclean : list of tuples
        [(sag, error), ...]
    bin_size : int
        Width of each sag bin.
    max_sag : int
        Maximum sag to consider.
    
    Returns
    -------
    bin_centers : list
        Center of each sag bin.
    bin_means : list
        Mean error in each bin.
    """
    bins = np.arange(0, max_sag + bin_size, bin_size)  # bin edges
    binned_means = []
    bin_centers = []

    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_end = bins[i+1]
        bin_center = (bin_start + bin_end)/2

        # select points in this bin
        values_in_bin = [y for x,y in Lclean if bin_start <= x < bin_end]
        if values_in_bin:
            mean_val = np.mean(values_in_bin)
        else:
            mean_val = np.nan  # or 0 if you prefer

        binned_means.append(mean_val)
        bin_centers.append(bin_center)

    return bin_centers, binned_means


def par_cat_error(P1, P3, coefs, cat_func, delta=1e-4):
    """
    Compute mean, max, total vertical error between a parabola and a catenary function.
    
    Parameters
    ----------
    P1, P3 : array-like
        Endpoints of the parabola.
    coefs : tuple/list
        Parabola coefficients (A, B, C) for y = Ax^2 + Bx + C
    cat_func : function
        Catenary function y = f(x)
    delta : float
        Sampling step along x-axis.
    
    Returns
    -------
    dict with mean, max, total error
    """
    A, B, C = coefs
    x1, _ = P1
    x3, _ = P3

    xs = np.arange(x1, x3 + delta, delta)
    errors = []

    for x in xs:
        y_par = A*x**2 + B*x + C
        y_cat = cat_func(x)
        errors.append(abs(y_par - y_cat))

    errors = np.array(errors)
    return {"mean": np.mean(errors), "max": np.max(errors), "total": np.sum(errors)}


from scipy.interpolate import interp1d



def fill_missing_bins(bin_centers, bin_means):
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    
    # If first or last value is nan, fill it with nearest non-nan
    if np.isnan(bin_means[0]):
        first_valid = np.argmax(~np.isnan(bin_means))
        bin_means[0:first_valid] = bin_means[first_valid]
    if np.isnan(bin_means[-1]):
        last_valid = len(bin_means) - 1 - np.argmax(~np.isnan(bin_means[::-1]))
        bin_means[last_valid+1:] = bin_means[last_valid]
    
    valid = ~np.isnan(bin_means)
    
    if valid.sum() < 2:
        return bin_centers, bin_means
    
    # Use cubic if enough points, else linear
    kind = 'cubic' if valid.sum() >= 4 else 'linear'
    
    interp_func = interp1d(bin_centers[valid], bin_means[valid], kind=kind, fill_value="extrapolate")
    smoothed = interp_func(bin_centers)
    
    return bin_centers, smoothed

if __name__ == "__main__":

    simulate(show_img=False, delta=10**-2)

    # plot_parabolas()

    # area_comparison(delta=10**-2)
