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

import random
import unittest

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))


def TensDotProf(a,b,inds,reps):
    for n in range(reps):
        r=snp.tensordot(a,b,(inds,inds),ignore_qflow=True)





parser = argparse.ArgumentParser('tensordotprof.py')
parser.add_argument('--N', help='number of different quantum number sectors per leg (10)' ,type=int,default=10)
parser.add_argument('--rank', help='tensor ranks' ,type=int,default=3)
parser.add_argument('--dim', help='dimension of each quantum number sector on the legs (100)' ,type=int,default=100)
parser.add_argument('--reps', help='number of repetitions of the tensordot contraction' ,type=int,default=1)
parser.add_argument('--nlegs', help='number of legs to be contracted (rank)' ,type=int)

args=parser.parse_args()
if args.nlegs==None:
    nlegs=args.rank
elif args.nlegs>args.rank:
    print('cannot contract more than "rank" legs')
    nlegs=args.rank
<<<<<<< HEAD
else:
    nlegs=args.nlegs
=======
>>>>>>> master
rank=args.rank
N=args.N
dim=args.dim
kd={}

for k,d in zip(range(N),[dim]*N):
    kd[k]=d


I=[]

for n in range(rank):
    I.append(spt.TensorIndex(kd,1))

a=spt.SparseTensor.random(I)
b=spt.SparseTensor.random(I)
<<<<<<< HEAD

=======
inds=range(rank)
nlegs=args.nlegs
>>>>>>> master
inds=random.sample(range(rank),nlegs)
reps=args.reps
cProfile.run('TensDotProf(a,b,inds,reps)','tensdotprof')
