import csv
import numpy as np

from mpl_toolkits.mplot3d import axes3d

#from pyFIMTDD import FIMTDD as FIMTGD
#from Greedy_FIMTDD_LS import FIMTDD as gFIMTLS
# from umcFIMTDD_LS import FIMTDD as gFIMTLS
from pyFIMTDD import FIMTDD as FIMTGD
# from FIMTDD_LS import FIMTDD as FIMTLS
import matplotlib.pyplot as plt
import itertools
import time
from multiprocessing import Pool
import multiprocessing as mp
import progressbar as pb
import os
import sys
from DataGenerator import *
from Legendre_Test import data_provider
sys.setrecursionlimit(100000)

def get_Kiel_data():
    print("setting up data...")
    data = list()
    with open('KielClean.csv', 'r') as fp:
        w_day = 1.0
        prev_day = -1.0
        prev_time = 0.0
        r = csv.reader(fp, delimiter=',')
        data = list()
        for i in r:
            row = list()
            for j in i:
                row.append(float(j))
            if row == []:
                continue
            data.append(row)
    data = np.array(data)
    new_data = list()
    print('len(data): ', len(data))
    simulationSteps = len(data) - 100
    print("data loaded in RAM")
    # print simulationSteps
    for idx in range(simulationSteps):
        new_data.append(np.hstack((data[idx][5], data[idx + 48][5], data[idx + 72][5],
                                   data[idx + 84][5], data[idx + 90][5], data[idx + 93][5],
                                   data[idx + 94][5], data[idx + 95][5], data[idx + 96][3:])))
    print("data prepared for processing")
    data = np.array(new_data)
    simulationSteps = len(data)
    return data
    # s#elf.bar = progressbar.ProgressBar(max_value=simulationSteps)

def Kiel_Test(paramlist,show,val):
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    # fimtls=FIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    # gfimtls=gFIMTLS(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[5])
    cumLossgd  =[0]
    # cumLossls  =[0]
    # cumLossgls =[0]

    if True:
        data = get_Kiel_data()
        c = 0

        # data_length = 100000
        data_length = 10
        # data_length = len(data)

        for i in range(data_length):
            c += 1
            print(str(c)+'/'+str(data_length))
            input = data[i][8:10]
            #target = data[1][i] + (np.random.uniform() - 0.5) * 0.2
            target = data[i][10]

            if i > -1:
                cumLossgd.append(cumLossgd[-1] + np.fabs(target - fimtgd.eval_and_learn(np.array(input), target)))
                # cumLossls.append(cumLossls[-1] + np.fabs(target - fimtls.eval_and_learn(np.array(input), target)))
                # cumLossgls.append(cumLossgls[-1] + np.fabs(target - gfimtls.eval_and_learn(np.array(input), target)))
            else:
                #warm start
                fimtgd.eval_and_learn(np.array(input), target)
                # fimtls.eval_and_learn(np.array(input), target)
                # gfimtls.eval_and_learn(np.array(input), target)
            #plt.scatter(x=x,y=y)
            #plt.show()
            if show:
                f=plt.figure()
                plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
                f.hold(True)
                # plt.plot(cumLossls[1:], label="Filter Loss")
               #avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
                #plt.plot(avglossgd,label="Average GD Loss")
                #plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
                # plt.title("CumLoss Ratio:"+str(min(cumLossgd[-1],cumLossls[-1])/max(cumLossgd[-1],cumLossls[-1])))
                plt.legend()
                figname="g"+str(paramlist[0])+"_nmin"+str(paramlist[1])+"_al"+str(paramlist[2])+"_thr"+str(paramlist[3])\
                        + "_lr"+str(paramlist[4])+".png"
                plt.savefig(figname)
                #plt.show()
                f.clear()
            #print(i)
            #print(fimtgd.count_leaves())
            #print(fimtgd.count_nodes())
            # print(fimtgd)
        # return [cumLossgd,cumLossls,cumLossgls,val,paramlist]
        return [cumLossgd,val,paramlist]

if __name__ == '__main__':
    # global result_list

    numberoftests =  1
    # result_list = [None]*numberoftests
    paramlist = [2.0, 10, 0.001, 50, 0.75, 5]
    results = Kiel_Test(paramlist,False,None)

    print(results)