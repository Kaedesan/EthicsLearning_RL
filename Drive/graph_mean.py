import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='print graph')

parser.add_argument('--fnh', type=str, default='null',
                    help='file of the negative human performance')
parser.add_argument('--fph', type=str, default='null',
                    help='file of the positive human performance')
parser.add_argument('--fmh', type=str, default='null',
                    help='file of the mixed human performance')
parser.add_argument('--fema', type=str, default='null',
                    help='file of the ethic mixed agent performance')
parser.add_argument('--fena', type=str, default='null',
                    help='file of the ethic negative agent performance')
parser.add_argument('--fepa', type=str, default='null',
                    help='file of the ethic positive agent performance')
parser.add_argument('--fca', type=str, default='null',
                    help='file of the classical agent performance')
parser.add_argument('--t', type=str, default='null',
                    help='parameter studied')
parser.add_argument('--nb_eps', type=int, default= 1000,
                    help='number of episodes')
parser.add_argument('--id', type=str, default= 'null',
                    help='identify the experiment')
parser.add_argument('--nb_exp', type=int, default= 10,
                    help='number of experiments')
parser.add_argument('--first_exp', type=int, default= 0,
                    help='index of the first experiment')

args = parser.parse_args()

printNH = False
printPH = False
printMH = False
printENA = False
printEPA = False
printEMA = False
printCA = False

x_axis = list(range(1,args.nb_eps+1))

nhlist = [0 for i in x_axis]
phlist = [0 for i in x_axis]
mhlist = [0 for i in x_axis]
enalist = [0 for i in x_axis]
epalist = [0 for i in x_axis]
emalist = [0 for i in x_axis]
calist = [0 for i in x_axis]

for exp in range(args.first_exp, args.first_exp + args.nb_exp):

    if args.fmh != 'null':
        mhdata = pd.read_csv('./record/'+str(exp)+'_'+args.fmh, header=None).to_dict()
        for i in x_axis:
            mhlist[i-1] += mhdata[0][i]
        printMH = True

    if args.fnh != 'null':
        nhdata = pd.read_csv('./record/'+str(exp)+'_'+args.fnh, header=None).to_dict()
        for i in x_axis:
            nhlist[i-1] += nhdata[0][i]
        printNH = True

    if args.fph != 'null':
        phdata = pd.read_csv('./record/'+str(exp)+'_'+args.fph, header=None).to_dict()
        for i in x_axis:
            phlist[i-1] += phdata[0][i]
        printPH = True

    if args.fema != 'null':
        emadata = pd.read_csv('./record/'+str(exp)+'_'+args.fema, header=None).to_dict()
        for i in x_axis:
            emalist[i-1] += emadata[0][i]
        printEMA = True

    if args.fepa != 'null':
        epadata = pd.read_csv('./record/'+str(exp)+'_'+args.fepa, header=None).to_dict()
        for i in x_axis:
            epalist[i-1] += epadata[0][i]
        printEPA = True

    if args.fena != 'null':
        enadata = pd.read_csv('./record/'+str(exp)+'_'+args.fena, header=None).to_dict()
        for i in x_axis:
            enalist[i-1] += enadata[0][i]
        printENA = True

    if args.fca != 'null':
        cadata = pd.read_csv('./record/'+str(exp)+'_'+args.fca, header=None).to_dict()
        for i in x_axis:
            calist[i-1] += cadata[0][i]
        printCA = True


plt.figure(figsize=(14,10))

if printNH == True:
    for i in x_axis:
        nhlist[i-1] = nhlist[i-1]/args.nb_exp
    plt.plot(x_axis , nhlist, c='b', label="human negative policy" )

if printPH == True:
    for i in x_axis:
        phlist[i-1] = phlist[i-1]/args.nb_exp
    plt.plot(x_axis , phlist, c='green', label="human positive policy" )

if printMH == True:
    for i in x_axis:
        mhlist[i-1] = mhlist[i-1]/args.nb_exp
    plt.plot(x_axis , mhlist, c='purple', label="human mixed policy" )

if printENA == True:
    for i in x_axis:
        enalist[i-1] = enalist[i-1]/args.nb_exp
    plt.plot(x_axis , enalist, c='r', label="ethical negative agent policy" )

if printEPA == True:
    for i in x_axis:
        epalist[i-1] = epalist[i-1]/args.nb_exp
    plt.plot(x_axis , epalist, c='yellow', label="ethical positive agent policy" )

if printEMA == True:
    for i in x_axis:
        emalist[i-1] = emalist[i-1]/args.nb_exp
    plt.plot(x_axis , emalist, c='orange', label="ethical mixed agent policy" )

if printCA == True:
    for i in x_axis:
        calist[i-1] = calist[i-1]/args.nb_exp
    plt.plot(x_axis , calist, c='black', label="classical agent policy" )

plt.legend()
plt.xlabel("Number of episodes")
plt.ylabel("score for " + args.t)
plt.title("Comparing the score of the different policies for " + args.t)

plt.savefig('./images/'+args.id+'_mean_comparison_'+args.t+'.png')
