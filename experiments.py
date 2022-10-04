import matplotlib.pyplot as plt
import random


def get_experiments(N, side):

    experiments = []

    while len(experiments) < N:

        xP1= round(random.uniform(0, side/2), 2) 
        yP1= round(random.uniform(side/2, side), 2)
        
        xP3= round(random.uniform(side/2, side), 2)
        yP3= round(random.uniform(side/2, side), 2)
        
        xP2= round(random.uniform(xP1, xP3), 2)
        yP2= round(random.uniform(0, side/2), 2)
        
        d = {"A" : [xP1, yP1], "B" : [xP2, yP2], "C" : [xP3, yP3]}
        experiments.append(d)

    return experiments



if __name__ == "__main__":

    result = get_experiments(2, 10)

    print(result)
