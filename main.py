import csv
import numpy as np

from pyFIMTDD import FIMTDD as FIMTGD
from FIMTDD_LS import FIMTDD as FIMTLS
from umcFIMTDD_LS import FIMTDD as gFIMTLS

import datetime
import os

# import sys
# sys.setrecursionlimit(100000)

def get_data():
    print("setting up data...")
    data = list()
    
    with open(dir_path + '/' +training_file_name, 'r') as fp:
        # w_day = 1.0
        # prev_day = -1.0
        # prev_time = 0.0
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
    print('training_data length: ', len(data))
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

def Kiel_Test(paramlist,show):
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    
    cumLossgd = []

    if True:
        c = 0

        for i in range(training_data_length):
            c += 1
            print(str(c)+'/'+str(training_data_length))

            input = training_data[i][8:10]
            target = training_data[i][10]

            if i > -1:
                cumLossgd.append(np.fabs(target - fimtgd.eval_and_learn(np.array(input), target)))
            else:
                #warm start
                fimtgd.eval_and_learn(np.array(input), target)

            #plt.scatter(x=x,y=y)
            #plt.show()
            # if show:
            #     f=plt.figure()
            #     plt.plot(cumLossgd[1:], label="Gradient Descent Loss")
            #     f.hold(True)
            #     # plt.plot(cumLossls[1:], label="Filter Loss")
            #    #avglossgd=np.array([cumLossgd[-1]/len(cumLossgd)]*len(cumLossgd))
            #     #plt.plot(avglossgd,label="Average GD Loss")
            #     #plt.plot([cumLossls[-1]/len(cumLossls)]*len(cumLossls), label="Average Filter Loss")
            #     # plt.title("CumLoss Ratio:"+str(min(cumLossgd[-1],cumLossls[-1])/max(cumLossgd[-1],cumLossls[-1])))
            #     plt.legend()
            #     figname="g"+str(paramlist[0])+"_nmin"+str(paramlist[1])+"_al"+str(paramlist[2])+"_thr"+str(paramlist[3])\
            #             + "_lr"+str(paramlist[4])+".png"
            #     plt.savefig(figname)
            #     #plt.show()
            #     f.clear()

        avglossgd = sum(cumLossgd)/len(cumLossgd)
        return [cumLossgd, avglossgd, fimtgd]

def traverse_tree(root):
    global file_writer
    global leaf_count

    isLeaf = root.isLeaf
    if isLeaf:
        # print('isLeaf: ', isLeaf)
        # print(root.model)
        file_writer.write('linear model coefficients on leaf: ' + str(root.model.w) + '\n')
        file_writer.write('\n')
        leaf_count = leaf_count + 1
    else:
        file_writer.write('splitting feature index (key_dim): ' + str(root.key_dim) + '\n')
        file_writer.write('splitting value (key): ' + str(root.key) + '\n')
        file_writer.write('\n')
        
        if root.left is not None:
            file_writer.write('has left child:\n')
            traverse_tree(root.left)
        
        if root.right is not None:
            file_writer.write('has right child:\n')
            traverse_tree(root.right)
    
def print_tree(tree, average_loss):
    print('printing tree...')
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tree_file_name = dir_path + '/tree_' + training_file_name + '_'+ str(timestamp) + '.txt'

    global file_writer
    file_writer = open(tree_file_name,'w')
    
    file_writer.write('average_loss: ' + str(average_loss) + '\n')
    file_writer.write('\n')

    global leaf_count
    leaf_count = 0

    root = tree.root
    traverse_tree(root)

    file_writer.write('leaf_count: ' + str(leaf_count) + '\n')
    file_writer.close()
    print('Tree printed in file: ', tree_file_name)
        
if __name__ == '__main__':
    global training_file_name
    training_file_name = 'KielClean.csv'
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    training_data = get_data()

    # training_data_length = 3000
    # training_data_length = 100000
    training_data_length = len(training_data)

    paramlist = [0.25, 10, 0.001, 10, 0.005, 2]
    results = Kiel_Test(paramlist,False)

    average_loss = results[1]
    
    tree = results[-1]
    print('tree: ', tree)

    print_tree(tree, average_loss)
