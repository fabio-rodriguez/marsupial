import json
import matplotlib.pyplot as plt
import random


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


if __name__ == "__main__":

    result = get_experiments(10, 35)

    print(result)

    
