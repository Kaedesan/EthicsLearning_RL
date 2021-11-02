import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='print graph')

parser.add_argument('--fcol_one', type=str, default='null',
                    help='file of the colision perf for agent 1')
parser.add_argument('--fcol_two', type=str, default='null',
                    help='file of the colision perf for agent 2')
parser.add_argument('--fcats_one', type=str, default='null',
                    help='file of the cat hits perf for agent 1')
parser.add_argument('--fcats_two', type=str, default='null',
                    help='file of the cat hits perf for agent 2')
parser.add_argument('--felders_one', type=str, default='null',
                    help='file of the elders saved perf for agent 1')
parser.add_argument('--felders_two', type=str, default='null',
                    help='file of the elders saved perf for agent 2')
#parser.add_argument('--t', type=str, default='null',help='parameter studied')
parser.add_argument('--fp', type=str, default='null',help='name of the first policy')
parser.add_argument('--sp', type=str, default='null',help='name of the second policy')
parser.add_argument('--nb_eps', type=int, default= 1000,
                    help='number of episodes')
parser.add_argument('--nb_steps', type=int, default= 300,
                    help='number of episodes')
parser.add_argument('--id', type=str, default= 'null',
                    help='identify the experiment')
parser.add_argument('--nb_exp', type=int, default= 10,
                    help='number of experiments')
parser.add_argument('--first_exp', type=int, default= 0,
                    help='index of the first experiment')

args = parser.parse_args()

printCOLO = False
printCOLT = False
printCATSO = False
printCATST = False
printELDO = False
printELDT = False

x_axis = list(range(args.nb_steps+1))

colo_list = [0 for i in x_axis]
colt_list = [0 for i in x_axis]
catso_list = [0 for i in x_axis]
catst_list = [0 for i in x_axis]
eldo_list = [0 for i in x_axis]
eldt_list = [0 for i in x_axis]

for exp in range(args.first_exp, args.first_exp + args.nb_exp):

    if args.fcol_one != 'null':
        colo_data = pd.read_csv('./record_timesteps/'+str(exp)+'_'+args.fcol_one, header=None).to_dict()
        for i in x_axis:
            colo_list[i] += colo_data[0][i]
        printCOLO = True

    if args.fcol_two != 'null':
        colt_data = pd.read_csv('./record_timesteps/'+str(exp)+'_'+args.fcol_two, header=None).to_dict()
        for i in x_axis:
            colt_list[i] += colt_data[0][i]
        printCOLT = True

    if args.fcats_one != 'null':
        catso_data = pd.read_csv('./record_timesteps/'+str(exp)+'_'+args.fcats_one, header=None).to_dict()
        for i in x_axis:
            catso_list[i] += catso_data[0][i]
        printCATSO = True

    if args.fcats_two != 'null':
        catst_data = pd.read_csv('./record_timesteps/'+str(exp)+'_'+args.fcats_two, header=None).to_dict()
        for i in x_axis:
            catst_list[i] += catst_data[0][i]
        printCATST = True

    if args.felders_one != 'null':
        eldo_data = pd.read_csv('./record_timesteps/'+str(exp)+'_'+args.felders_one, header=None).to_dict()
        for i in x_axis:
            eldo_list[i] += eldo_data[0][i]
        printELDO = True

    if args.felders_two != 'null':
        eldt_data = pd.read_csv('./record_timesteps/'+str(exp)+'_'+args.felders_two, header=None).to_dict()
        for i in x_axis:
            eldt_list[i] += eldt_data[0][i]
        printELDT = True

plt.figure(figsize=(14,10))

if printCOLO == True:
    for i in x_axis:
        if colo_list[i] != 0:
            colo_list[i] = colo_list[i]/args.nb_exp
    plt.plot(x_axis , colo_list, c='b', label="collisions of " + args.fp )

if printCOLT == True:
    for i in x_axis:
        if colt_list[i] != 0:
            colt_list[i] = colt_list[i]/args.nb_exp
    plt.plot(x_axis , colt_list, c='green', label="collisions of " + args.sp )

if printCATSO == True:
    for i in x_axis:
        if catso_list[i] != 0:
            catso_list[i] = catso_list[i]/args.nb_exp
    plt.plot(x_axis , catso_list, c='purple', label="cats hit of " + args.fp )

if printCATST == True:
    for i in x_axis:
        if catst_list[i] != 0:
            catst_list[i] = catst_list[i]/args.nb_exp
    plt.plot(x_axis , catst_list, c='r', label="cats hit of " + args.sp )

if printELDO == True:
    for i in x_axis:
        if eldo_list[i] != 0:
            eldo_list[i] = eldo_list[i]/args.nb_exp
    plt.plot(x_axis , eldo_list, c='yellow', label="elders saved of " + args.fp )

if printELDT == True:
    for i in x_axis:
        if eldt_list[i] != 0:
            eldt_list[i] = eldt_list[i]/args.nb_exp
    plt.plot(x_axis , eldt_list, c='orange', label="elders saved of " + args.sp )

plt.legend()
plt.xlabel("Number of steps")
plt.ylabel("Score")
plt.title("Comparing the score of the different policies")

plt.savefig('./images_timesteps/'+args.id+'_mean_comparison_global.png')
