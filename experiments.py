import json
import matplotlib.pyplot as plt
import numpy as np
import random

from get_parables import touch_the_ground, get_par_from_3points, approx_parable_length


def get_experiments(N, side, path="data/experiments.json"):

    experiments = []

    while len(experiments) < N:

        xP1= round(random.uniform(0, side/2), 2) 
        yP1= round(random.uniform(0, side/2), 2)
        
        xP3= round(random.uniform(side/2, side), 2)
        yP3= round(random.uniform(yP1, side), 2)
        
        Xrange = (xP3-xP1)/4 # condition for having not too long catenaries
        xP2= round(random.uniform(xP1+Xrange, xP3-Xrange), 2)
        yP2= round(random.uniform(0, yP1), 2)
        
        d = {"A" : [xP1, yP1], "B" : [xP2, yP2], "C" : [xP3, yP3]}
        experiments.append(d)

    f = open(path, "w")
    f.write(json.dumps(experiments))
    f.close()

    return experiments


def get_experiments_above_zero(N, side, path="data/experiments.json"):

    experiments = []

    while len(experiments) < N:

        xP1= 0
        yP1= 1
        
        xP3= round(random.uniform(side/2, side), 2)
        yP3= round(random.uniform(yP1, side), 2)
        
        Xrange = (xP3-xP1)/4 # condition for having not too long catenaries
        xP2= round(random.uniform(xP1+Xrange, xP3-Xrange), 2)
        yP2= round(random.uniform(0, yP3*xP2/xP3), 2)
        
        d = {"A" : [xP1, yP1], "B" : [xP2, yP2], "C" : [xP3, yP3]}
        experiments.append(d)

    f = open(path, "w")
    f.write(json.dumps(experiments))
    f.close()

    return experiments


def get_experiments3(N, side, Lmax=50, path="data/experiments.json"):

    experiments = []
    while len(experiments) <= N:
        print(len(experiments))
        xP1, yP1 = 0, 1
        
        xP3= round(random.uniform(side/3, side), 2)
        yP3= round(random.uniform(yP1, side), 2)
        
        count, maxcount = 0, 50
        while True:
            count += 1
            if count > maxcount:
                break

            xP2= round(random.uniform(xP1, xP3), 2)
            prop = (xP2-xP1)/(xP3-xP1)
            yP2= round(random.uniform(yP1, (yP3 - yP1)*prop), 2)

            # if (yP3 - yP1)*prop < (yP2 - yP1):
            #     continue

            coefs, _ = get_par_from_3points(xP1, yP1, xP2, yP2, xP3, yP3)       
            A, B, C = coefs
            f = lambda x: A*x**2 + B*x + C

            length = approx_parable_length([xP1, yP1], [xP2, yP2], [xP3, yP3], A, B, C)
            if not touch_the_ground(f, [int(xP1)+1, int(xP3)-1], delta=10**-3) and length < Lmax:
                print(length)
                break
        
        if count > maxcount:
            continue

        d = {"A" : [xP1, yP1], "B" : [xP2, yP2], "C" : [xP3, yP3]}
        experiments.append(d)

    f = open(path, "w")
    f.write(json.dumps(experiments))
    f.close()

    return experiments


if __name__ == "__main__":

    # result = get_experiments(10, 35)

    # result = get_experiments_above_zero(100, 35)
    
    result = get_experiments3(100, 40)

    print(result)

