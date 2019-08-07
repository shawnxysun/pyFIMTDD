import csv
import numpy as np

from pyFIMTDD import FIMTDD as FIMTGD
from FIMTDD_LS import FIMTDD as FIMTLS
from umcFIMTDD_LS import FIMTDD as gFIMTLS

import datetime
import os

import sys
sys.setrecursionlimit(200000)

def parse_training_data():
    print("setting up data...")
    data = list()
    hasHeader = True
    
    with open(training_file_path + '/' +training_file_name, 'r') as fp:
        # w_day = 1.0
        # prev_day = -1.0
        # prev_time = 0.0
        r = csv.reader(fp, delimiter=',')
        data = list()
        for i in r:

            # skip header
            if hasHeader == True:
                hasHeader = False
                continue

            row = list()
            for j in i:
                row.append(float(j))
            if row == []:
                continue
            data.append(row)
    data = np.array(data)
    print('training_data length: ', len(data))

    # new_data = list()
    # simulationSteps = len(data) - 100
    # print("data loaded in RAM")
    # # print simulationSteps
    # for idx in range(simulationSteps):
    #     new_data.append(np.hstack((data[idx][5], data[idx + 48][5], data[idx + 72][5],
    #                                data[idx + 84][5], data[idx + 90][5], data[idx + 93][5],
    #                                data[idx + 94][5], data[idx + 95][5], data[idx + 96][3:])))
    # print("data prepared for processing")
    # data = np.array(new_data)
    # simulationSteps = len(data)
    return data

def build_tree(paramlist):
    fimtgd=FIMTGD(gamma=paramlist[0], n_min = paramlist[1], alpha=paramlist[2], threshold=paramlist[3], learn=paramlist[4])
    
    cumLossgd = []

    if True:
        c = 0

        for i in range(training_data_length):
            c += 1
            print(str(c)+'/'+str(training_data_length))

            target = training_data[i][0]
            input = training_data[i][1:]

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
    current_directory = os.path.dirname(os.path.realpath(__file__))
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tree_file_name = current_directory + '/tree_' + training_file_name + '_'+ str(timestamp) + '.txt'

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
    # training_file_name = 'KielClean.csv'
    # training_file_name = 'sportlogiq_data_pass_2019_07_22_08_54_17_small.csv'
    training_file_name = 'sportlogiq_data_pass_2019_07_22_08_54_17.csv'
    
    # training_file_path = os.path.dirname(os.path.realpath(__file__))
    training_file_path = '/cs/oschulte/xiangyus/csv_training_files'
    training_data = parse_training_data()

    training_data_length = len(training_data)
    # training_data_length = 1000
    # training_data_length = 100000
    
    # [gamma, n_min, alpha, threshold, learn]
    # :param gamma/delta:       hoefding-bound value
    # :param n_min:       minimum intervall for split and alt-tree replacement
    # :param alpha:       used for change detection
    # :param threshold:   threshold for change detection
    # :param learn:   learning rate
    paramlist = [0.01, 200, 0.005, 50, 0.01]

    results = build_tree(paramlist)

    average_loss = results[1]
    
    tree = results[-1]
    print('tree: ', tree)

    print_tree(tree, average_loss)
