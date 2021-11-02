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

args = parser.parse_args()

printNH = False
printPH = False
printMH = False
printENA = False
printEPA = False
printEMA = False
printCA = False


if args.fmh != 'null':
    mhdata = pd.read_csv(args.fmh, header=None).to_dict()
    printMH = True

if args.fnh != 'null':
    nhdata = pd.read_csv(args.fnh, header=None).to_dict()
    printNH = True

if args.fph != 'null':
    phdata = pd.read_csv(args.fph, header=None).to_dict()
    printPH = True

if args.fema != 'null':
    emadata = pd.read_csv(args.fema, header=None).to_dict()
    printEMA = True

if args.fepa != 'null':
    epadata = pd.read_csv(args.fepa, header=None).to_dict()
    printEPA = True

if args.fena != 'null':
    enadata = pd.read_csv(args.fena, header=None).to_dict()
    printENA = True

if args.fca != 'null':
    cadata = pd.read_csv(args.fca, header=None).to_dict()
    printCA = True

x_axis = list(range(1,args.nb_eps+1))
plt.figure(figsize=(14,10))
if printNH == True:
    nhlist = [nhdata[0][i] for i in x_axis]
    plt.plot(x_axis , nhlist, c='b', label="human negative policy" )
if printPH == True:
    phlist = [phdata[0][i] for i in x_axis]
    plt.plot(x_axis , phlist, c='green', label="human positive policy" )
if printMH == True:
    mhlist = [mhdata[0][i] for i in x_axis]
    plt.plot(x_axis , mhlist, c='purple', label="human mixed policy" )
if printENA == True:
    enalist = [enadata[0][i] for i in x_axis]
    plt.plot(x_axis , enalist, c='r', label="ethical negative agent policy" )
if printEPA == True:
    epalist = [epadata[0][i] for i in x_axis]
    plt.plot(x_axis , epalist, c='yellow', label="ethical positive agent policy" )
if printEMA == True:
    emalist = [emadata[0][i] for i in x_axis]
    plt.plot(x_axis , emalist, c='orange', label="ethical mixed agent policy" )
if printCA == True:
    calist = [cadata[0][i] for i in x_axis]
    plt.plot(x_axis , calist, c='black', label="classical agent policy" )

plt.legend()
plt.xlabel("Number of episodes")
plt.ylabel("score for " + args.t)
plt.title("Comparing the score of the different policies for " + args.t)

plt.savefig('./images/'+args.id+'_comparison_'+args.t+'.png')
