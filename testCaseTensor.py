#!/usr/bin/env python

#cmps program for ground state calculations of the inhomogeneous Lieb-Liniger model in the thermodynamic limit
#in the following comments, [,] is the commutator and {,} is the anti commutator of operators
#the following code employes the left gauge fixing through out, which results in l=11. Hence, l
#does not appear in the code

import sys
#sys.path.append('../lib_mps')
#sys.path.append('../symmat')
import numpy as np
import sparsenumpy as snp
import SparseTensor as spt
import qutilities as utils
import random
import unittest

#from keyclass import AbelianKey
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))
verbose=True



class TensorInitTests(unittest.TestCase):
    def setUp(self):
        self.rank=4
        self.outind=random.sample(range(self.rank),1)[0]
        I1=spt.TensorIndex({10:4,3:7,4:3,9:12},1)
        I2=spt.TensorIndex({2:2,1:6,4:2,6:11},1)
        I3=spt.TensorIndex({2:2,1:6,4:2,6:11},1)
        I4=spt.TensorIndex({14:6,12:3,18:2,10:11,21:11},-1)                        
        self.I=[I1,I2,I3,I4]
        self.eps=1E-10
        
    def test_random(self):
        ten=spt.SparseTensor.random(self.I)
    def test_zeros(self):
        ten=spt.SparseTensor.zeros(self.I)
        for t in ten._tensor.values():
            self.assertTrue(np.linalg.norm(t)<self.eps)
    def test_ones(self):
        ten=spt.SparseTensor.ones(self.I)
        for t in ten._tensor.values():
            self.assertTrue(np.linalg.norm(t-np.ones(t.shape).astype(t.dtype))<self.eps)



    def test_random_like(self):
        ten=spt.SparseTensor.random(self.I)
        ten2=spt.SparseTensor.random_like(ten)
    def test_zeros_like(self):
        ten=spt.SparseTensor.ones(self.I)
        ten2=spt.SparseTensor.ones_like(ten)
    def test_ones_like(self):
        ten=spt.SparseTensor.zeros(self.I)
        ten2=spt.SparseTensor.zeros_like(ten)
        

class TensorLinalgTests(unittest.TestCase):
    def setUp(self):
        self.N=10
        self.rank=4
        keys=[]
        outind=random.sample(range(self.rank),1)[0]
        #s=random.sample(range(10000),1)[0]
        #s=9131
        #random.seed(s)
        for n in range(self.N):
            key=random.sample(range(10),self.rank-1)
            key.insert(outind,sum(key))
            keys.append(tuple(key))

        Ds=[dict() for n in range(self.rank)]
        for n in range(self.rank):
            for k in keys:
                Ds[n][k[n]]=random.sample(range(2,4),1)[0]

        values=[]
        for k in keys:
            size=tuple([])
            for n in range(self.rank):
                size+=tuple([Ds[n][k[n]]])
            values.append(np.random.rand(*size)+1j*np.random.rand(*size))
            #values.append(np.random.rand(*size))            
            
        l=[1]*self.rank
        l[outind]=-1
        qflow=tuple(l)
        self.tens=spt.SparseTensor(keys=keys,values=values,Ds=Ds,qflow=qflow,mergelevel=None,dtype=complex)
        self.eps=1E-12


    def test_eye(self):
        ind=self.rank-1
        eye1=self.tens.__eye__(ind,0)
        eye2=self.tens.__eye__(ind,1)  
        r1=snp.tensordot(self.tens,eye1,([ind],[0]))
        r2=snp.tensordot(self.tens,eye2,([ind],[1]))

        diff1=set(r1.__keys__()).difference(self.tens.__keys__())
        diff2=set(self.tens.__keys__()).difference(r1.__keys__())
        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)

        diff1=set(r2.__keys__()).difference(self.tens.__keys__())
        diff2=set(self.tens.__keys__()).difference(r2.__keys__())
        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)

        for k in r1._tensor.keys():
            self.assertTrue(np.linalg.norm(r1[k]-self.tens[k])/utils.prod(self.tens[k].shape)<self.eps)
        for k in r2._tensor.keys():
            self.assertTrue(np.linalg.norm(r2[k]-self.tens[k])/utils.prod(self.tens[k].shape)<self.eps)


    def test_random(self):
        ind=self.rank-1
        rand1=self.tens.__random__(ind,0)
        rand2=self.tens.__random__(ind,1)        
        r1=snp.tensordot(self.tens,rand1,([ind],[0]))
        r2=snp.tensordot(self.tens,rand2,([ind],[1]))
        
        diff1=set(r1.__keys__()).difference(self.tens.__keys__())
        diff2=set(self.tens.__keys__()).difference(r1.__keys__())
        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)

        diff1=set(r2.__keys__()).difference(self.tens.__keys__())
        diff2=set(self.tens.__keys__()).difference(r2.__keys__())
        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)

            
    def test_tensordot(self):
        inds=random.sample(range(self.rank),random.sample(range(1,self.rank+1),1)[0])
        r=snp.tensordot(self.tens,self.tens,(inds,inds),ignore_qflow=True)

        full=self.tens.__tondarray__()
        rfull=np.tensordot(full,full,(inds,inds))
        self.assertTrue(np.linalg.norm(rfull-r.__tondarray__())<1E-10)
        shapes=r.__shapes__()
        for k in shapes:
            self.assertTrue(tuple(shapes[k])==r[k].shape)
            

    def test_tensordot_merged(self):
        num=random.sample(range(self.rank-1),1)[0]
        inds=random.sample(range(self.rank-1),random.sample(range(1,self.rank),1)[0])
        tens=snp.mergeSingle(self.tens,[num,num+1])
        r=snp.tensordot(tens,tens,(inds,inds),ignore_qflow=True)

        full=tens.__tondarray__()
        rfull=np.tensordot(full,full,(inds,inds))
        self.assertTrue(np.linalg.norm(rfull-r.__tondarray__())<1E-10)
        shapes=r.__shapes__()        
        for k in shapes:
            self.assertTrue(utils.prod(utils.flatten(tuple(shapes[k])))==utils.prod(r[k].shape))
        
    
    def test_tensordot_merged_2(self):
        N=5
        rank=8
        keys=[]
        outind=random.sample(range(rank),1)[0]
    
        for n in range(N):
            key=random.sample(range(20),rank-1)
            key.insert(outind,sum(key))
            keys.append(tuple(key))
    
        Ds=[dict() for n in range(rank)]
        for n in range(rank):
            for k in keys:
                Ds[n][k[n]]=random.sample(range(2,8),1)[0]
    
        values=[]
        for k in keys:
            size=tuple([])
            for n in range(rank):
                size+=tuple([Ds[n][k[n]]])
            values.append(np.random.rand(*size))
    
        l=[1]*rank
        l[outind]=-1
        qflow=tuple(l)
        tens=spt.SparseTensor(keys=keys,values=values,Ds=Ds,qflow=qflow,mergelevel=None,dtype=self.tens._dtype)
        eps=1E-12
    
        tens1=snp.merge(tens,[0,1],[3,4])
        tens2=snp.merge(tens1,[0,1],[3,4,5])
        inds=[1,2]
        r=snp.tensordot(tens2,tens2,(inds,inds),ignore_qflow=True)
        shapes=r.__shapes__()
        for k in shapes:
            self.assertTrue(utils.prod(utils.flatten(tuple(shapes[k])))==utils.prod(r[k].shape))
        
    def test_transpose(self):
        newinds=random.sample(range(self.rank),self.rank)
        transp=snp.transpose(self.tens,newinds)
        shapes=transp.__shapes__()
        for k in shapes:
            self.assertTrue(utils.prod(utils.flatten(tuple(shapes[k])))==utils.prod(transp[k].shape))

    
    def test_splitSingle(self):
        num=random.sample(range(self.rank-1),1)[0]
        tens=snp.mergeSingle(self.tens,[num,num+1])
        split=snp.splitSingle(tens,num)
        
        k1=sorted(split.__keys__())
        k2=sorted(self.tens.__keys__())
        self.assertTrue(k1==k2)
        r=split-self.tens
        self.assertTrue(r.__norm__()<1E-10)        
    
    def test_multisplitSingle(self):
        num=random.sample(range(self.rank-1),1)[0]
        tens=snp.mergeSingle(self.tens,[num,num+1])
        num2=random.sample(range(self.rank-2),1)[0]
        tens2=snp.mergeSingle(tens,[num2,num2+1])
        split=snp.splitSingle(tens2,num2)
        k1=sorted(split.__keys__())
        k2=sorted(tens.__keys__())
        self.assertTrue(k1==k2)
        r=split-tens
        self.assertTrue(r.__norm__()<1E-10)
        split2=snp.splitSingle(tens,num)
        k1=sorted(split2.__keys__())
        k2=sorted(self.tens.__keys__())
        self.assertTrue(k1==k2)

        r=split2-self.tens
        self.assertTrue(r.__norm__()<1E-10)
        
    def test_split(self):
        tens=snp.merge(self.tens,[0,1],[2,3])
        split=snp.split(tens,[0,1])
        k1=sorted(split.__keys__())
        k2=sorted(self.tens.__keys__())
        self.assertTrue(k1==k2)
        r=split-self.tens
        self.assertTrue(r.__norm__()<1E-10)
    
    def test_tensor_keys(self):
        num=random.sample(range(self.rank-1),1)[0]
        #tens=snp.mergeSingle(self.tens,tuple([num,num+1]))
        tens=snp.mergeSingle(self.tens,[num,num+1])
        N0=0
        for k in tens._keys[0]:
            N0+=len(tens._keys[0][k])
        for n in range(1,len(tens._keys)):
            N=0
            for k in tens._keys[n]:
                N+=len(tens._keys[n][k])
            self.assertTrue(N==N0)
        num2=random.sample(range(self.rank-2),1)[0]
        #tens2=snp.mergeSingle(tens,tuple([num2,num2+1]))
        tens2=snp.mergeSingle(tens,[num2,num2+1])
        split=snp.splitSingle(tens2,num2)
        N0=0
        for k in split._keys[0]:
            N0+=len(split._keys[0][k])
        for n in range(1,len(split._keys)):
            N=0
            for k in split._keys[n]:
                N+=len(split._keys[n][k])
            self.assertTrue(N==N0)
    
    def test_vectorize(self):
        vec,data=snp.vectorize(self.tens)
        tens=snp.tensorize(data,vec)
        self.assertTrue((tens-self.tens).__norm__()<1E-13)
    
        num=random.sample(range(self.rank-1),1)[0]
        #merged=snp.mergeSingle(self.tens,tuple([num,num+1]))
        merged=snp.mergeSingle(self.tens,[num,num+1])
        vec,data=snp.vectorize(merged)
        tens=snp.tensorize(data,vec)
        self.assertTrue((tens-merged).__norm__()<1E-13)
    
        split=snp.splitSingle(merged,num)
        vec,data=snp.vectorize(split)
        tens=snp.tensorize(data,vec)
        self.assertTrue((tens-split).__norm__()<1E-13)
        self.assertTrue((self.tens-split).__norm__()<1E-13)

    def test_norm(self):
        Z1=self.tens.__norm__()
        Z2=np.sqrt(list(snp.tensordot(self.tens,snp.conj(self.tens),(range(self.tens._rank),range(self.tens._rank)))._tensor.values())[0])
        self.assertTrue(np.abs(Z1-Z2)<self.eps)
        


class TensorTestUnaryAndBinaryOperations(unittest.TestCase):
    def setUp(self):
        self.N=10
        self.rank=4
        keys=[]
        outind=random.sample(range(self.rank),1)[0]
        #s=random.sample(range(10000),1)[0]
        #s=9131
        #random.seed(s)
        for n in range(self.N):
            key=random.sample(range(10),self.rank-1)
            key.insert(outind,sum(key))
            keys.append(tuple(key))

        Ds=[dict() for n in range(self.rank)]
        for n in range(self.rank):
            for k in keys:
                Ds[n][k[n]]=random.sample(range(2,4),1)[0]

        values=[]
        for k in keys:
            size=tuple([])
            for n in range(self.rank):
                size+=tuple([Ds[n][k[n]]])
            values.append(np.random.rand(*size)+1j*np.random.rand(*size))
            #values.append(np.random.rand(*size))            
            
        l=[1]*self.rank
        l[outind]=-1
        qflow=tuple(l)
        self.tens=spt.SparseTensor(keys=keys,values=values,Ds=Ds,qflow=qflow,mergelevel=None,dtype=complex)
        self.eps=1E-12


    def test_equ(self):
        t1=self.tens.__copy__()
        self.assertTrue(t1==self.tens)

    def test_neg(self):
        t1=-self.tens.__copy__()
        self.assertTrue((t1+self.tens)==spt.SparseTensor.zeros_like(t1))
        
    def test_iadd(self):
        t1=self.tens.__copy__()
        t1.__randomize__()
        t1array=t1.__tondarray__()        
        tiadd=t1.__copy__()
        t2=self.tens.__copy__()
        t2.__randomize__()
        t2array=t2.__tondarray__()                
        tiadd+=t2
        tadd=t1+t2
        taddarray=t1array+t2array
        self.assertTrue(np.linalg.norm(taddarray-tiadd.__tondarray__())<self.eps)        
        self.assertTrue(tadd==tiadd)
        

    def test_add(self):
        t1=self.tens.__copy__()
        t2=self.tens.__copy__()
        t3=self.tens.__copy__()
        t1.__randomize__()
        t2.__randomize__()
        t3.__randomize__()        
        a=np.random.rand(1)[0]
        b=np.random.rand(1)[0]
        t=t1*a+t2*b
        tarray=t.__tondarray__()
        tarray_=a*t1.__tondarray__()+b*t2.__tondarray__()
        self.assertTrue(np.linalg.norm(tarray-tarray_)<self.eps)



    def test_isub(self):
        t1=self.tens.__copy__()
        t1.__randomize__()
        t1array=t1.__tondarray__()        
        tisub=t1.__copy__()
        t2=self.tens.__copy__()
        t2.__randomize__()
        t2array=t2.__tondarray__()                
        tisub-=t2
        tsub=t1-t2
        tsubarray=t1array-t2array
        self.assertTrue(np.linalg.norm(tsubarray-tisub.__tondarray__())<self.eps)        
        self.assertTrue(tsub==tisub)
        
    def test_sub(self):
        t1=self.tens.__copy__()
        t2=self.tens.__copy__()
        t3=self.tens.__copy__()
        t1.__randomize__()
        t2.__randomize__()
        t3.__randomize__()        
        a=np.random.rand(1)[0]
        b=np.random.rand(1)[0]
        t=t1*a-t2*b
        tarray=t.__tondarray__()
        tarray_=a*t1.__tondarray__()-b*t2.__tondarray__()
        self.assertTrue(np.linalg.norm(tarray-tarray_)<self.eps)

        
class TestMatrixDecompositions(unittest.TestCase):
    def setUp(self):
        self.N=100
        self.rank=6
        keys=[]

        outind=random.sample(range(self.rank),1)[0]
        
        for n in range(self.N):
            key=random.sample(range(100),self.rank-1)
            key.insert(outind,sum(key))
            keys.append(tuple(key))

        Ds=[dict() for n in range(self.rank)]
        for n in range(self.rank):
            for k in keys:
                Ds[n][k[n]]=random.sample(range(2,10),1)[0]

        values=[]
        for k in keys:
            size=tuple([])
            for n in range(self.rank):
                size+=tuple([Ds[n][k[n]]])
            values.append(np.random.rand(*size)+1j*np.random.rand(*size))
            #values.append(np.random.rand(*size))

        l=[1]*self.rank
        l[outind]=-1
        qflow=tuple(l)
        keytoq=[dict() for n in range(self.rank)]
        for k in keys:
            for n in range(self.rank):
                keytoq[n][k[n]]=k[n]
                
        self.tens=spt.SparseTensor(keys=keys,values=values,Ds=Ds,qflow=qflow,keytoq=tuple(keytoq),mergelevel=None,dtype=complex)        
        self.assertTrue(list(self.tens.__charge__().values())==[0]*len(self.tens.__charge__().values()))
        self.eps=1E-12
        
    def test_svd(self):

        U,S,V=snp.svd(self.tens,[0,1,2],[3,4,5])

        Z=S.__norm__()
        self.tens/=Z
        S/=Z
        S.__squeeze__()

        US=snp.tensordot(U,S,([3],[0]))
        USV=snp.tensordot(US,V,([3],[0]))
        USV.__squeeze__()
        diff1=set(self.tens._tensor.keys()).difference(USV._tensor.keys())
        diff2=set(USV._tensor.keys()).difference(self.tens._tensor.keys())
        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)
        
        
        for k in self.tens._tensor.keys():
            self.assertTrue(np.linalg.norm(self.tens[k]-USV[k])/utils.prod(USV[k].shape)<self.eps)
        
        for k in USV._tensor.keys():
            self.assertTrue(np.linalg.norm(self.tens[k]-USV[k])/utils.prod(USV[k].shape)<self.eps)
        
        unitU=snp.tensordot(U,snp.conj(U),([0,1,2],[0,1,2]))
        for k in unitU._tensor.keys():
            self.assertTrue(np.linalg.norm(unitU[k]-np.eye(unitU[k].shape[0]))/utils.prod(unitU[k].shape)<self.eps)
        unitV=snp.tensordot(V,snp.conj(V),([1,2,3],[1,2,3]))
        for k in unitV._tensor.keys():
            self.assertTrue(np.linalg.norm(unitV[k]-np.eye(unitV[k].shape[0]))/utils.prod(unitV[k].shape)<self.eps)


    def test_qr(self):
        Q,R=snp.qr(self.tens,[0,1,2],[3,4,5])
        
        QR=snp.tensordot(Q,R,([3],[0]))
        QR.__squeeze__()
        diff1=set(self.tens.__keys__()).difference(QR.__keys__())
        diff2=set(QR.__keys__()).difference(self.tens.__keys__())

        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)

        for k in self.tens._tensor.keys():
            self.assertTrue(np.linalg.norm(self.tens[k]-QR[k])/utils.prod(QR[k].shape)<self.eps)

        for k in QR._tensor.keys():
            self.assertTrue(np.linalg.norm(self.tens[k]-QR[k])/utils.prod(QR[k].shape)<self.eps)
        
        unit=snp.tensordot(Q,snp.conj(Q),([0,1,2],[0,1,2]))
        for k in unit._tensor.keys():
            self.assertTrue(np.linalg.norm(unit[k]-np.eye(unit[k].shape[0]))/utils.prod(unit[k].shape)<self.eps)
            
    def test_qr_merged(self):
        merged=snp.merge(self.tens,[0,1],[4,5])
        Q,R=snp.qr(merged,[0,1],[2,3])
        #print
        #print 'merged:',
        #print 'qflow:',merged._qflow
        #print 'mergelevel:',merged._mergelevel
        #print 'charge:',merged.__charge__().values()
        #
        #print 'Q:'
        #print 'qflow:',Q._qflow
        #print 'mergelevel:',Q._mergelevel
        #print 'charge:',sorted(Q.__charge__().values())
        #print
        #
        #print 'R:'
        #print 'qflow:',R._qflow
        #print 'mergelevel:',R._mergelevel
        #print 'charge:',sorted(R.__charge__().values())
        #print 
        
        QR=snp.tensordot(Q,R,([2],[0]))
        QR.__squeeze__()
        diff1=set(merged.__keys__()).difference(QR.__keys__())
        diff2=set(QR.__keys__()).difference(merged.__keys__())

        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)

        for k in merged._tensor.keys():
            self.assertTrue(np.linalg.norm(merged[k]-QR[k])/utils.prod(QR[k].shape)<self.eps)

        for k in QR._tensor.keys():
            self.assertTrue(np.linalg.norm(merged[k]-QR[k])/utils.prod(QR[k].shape)<self.eps)
        
        unit=snp.tensordot(Q,snp.conj(Q),([0,1],[0,1]))
        for k in unit._tensor.keys():
            self.assertTrue(np.linalg.norm(unit[k]-np.eye(unit[k].shape[0]))/utils.prod(unit[k].shape)<self.eps)
            


    def test_svd_merged(self):
        merged=snp.merge(self.tens,[0,1],[4,5])
        U,S,V=snp.svd(merged,[0,1],[2,3])

        Z=S.__norm__()
        merged/=Z
        S/=Z
        S.__squeeze__()

        US=snp.tensordot(U,S,([2],[0]))
        USV=snp.tensordot(US,V,([2],[0]))
        USV.__squeeze__()
        diff1=set(merged._tensor.keys()).difference(USV._tensor.keys())
        diff2=set(USV._tensor.keys()).difference(merged._tensor.keys())
        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)
        
        
        for k in merged._tensor.keys():
            self.assertTrue(np.linalg.norm(merged[k]-USV[k])/utils.prod(USV[k].shape)<self.eps)
        
        for k in USV._tensor.keys():
            self.assertTrue(np.linalg.norm(merged[k]-USV[k])/utils.prod(USV[k].shape)<self.eps)
        
        unitU=snp.tensordot(U,snp.conj(U),([0,1],[0,1]))
        for k in unitU._tensor.keys():
            self.assertTrue(np.linalg.norm(unitU[k]-np.eye(unitU[k].shape[0]))/utils.prod(unitU[k].shape)<self.eps)
        unitV=snp.tensordot(V,snp.conj(V),([1,2],[1,2]))
        for k in unitV._tensor.keys():
            self.assertTrue(np.linalg.norm(unitV[k]-np.eye(unitV[k].shape[0]))/utils.prod(unitV[k].shape)<self.eps)


if __name__ == "__main__":
    suite0 = unittest.TestLoader().loadTestsFromTestCase(TensorInitTests)
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TensorTestUnaryAndBinaryOperations)    
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TensorLinalgTests)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestMatrixDecompositions)

    unittest.TextTestRunner(verbosity=2).run(suite0)    
    unittest.TextTestRunner(verbosity=2).run(suite1)
    unittest.TextTestRunner(verbosity=2).run(suite2)
    unittest.TextTestRunner(verbosity=2).run(suite3)    
