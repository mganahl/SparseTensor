#!/usr/bin/env python

#cmps program for ground state calculations of the inhomogeneous Lieb-Liniger model in the thermodynamic limit
#in the following comments, [,] is the commutator and {,} is the anti commutator of operators
#the following code employes the left gauge fixing through out, which results in l=11. Hence, l
#does not appear in the code

import sys,cProfile
import numpy as np
import sparsenumpy as snp
import SparseTensor as spt
import qutilities as utils
import argparse

import random,time
import unittest

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


def SVDProf(a,reps):
    for n in range(reps):
        U,S,V=snp.svd(a,[0,1],[2,3])





parser = argparse.ArgumentParser('tensordotprof.py')
parser.add_argument('--N', help='number of different quantum number sectors per leg (10)' ,type=int,default=10)
parser.add_argument('--dim', help='dimension of each quantum number sector on the legs (100)' ,type=int,default=100)
parser.add_argument('--reps', help='number of repetitions of the tensordot contraction' ,type=int,default=1)
parser.add_argument('--nlegs', help='number of legs to be contracted (rank)' ,type=int)

args=parser.parse_args()
rank=4
N=args.N
dim=args.dim
kd={}
dim=10
N=20
for k,d in zip(range(N),[dim]*N):
    kd[k]=d


I=[]

for n in range(rank-1):
    I.append(spt.TensorIndex(kd,1))
I.append(spt.TensorIndex(kd,-1))

a=spt.SparseTensor.random(I)
reps=args.reps
t1=time.time()
#U,S,V=snp.svd(a,[0,1],[2,3])
cProfile.run('SVDProf(a,reps)','svdprof')
t2=time.time()
print(t2-t1)

