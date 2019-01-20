#!/usr/bin/env python
#from __future__ import division
import timeit
import collections
import pandas as pd
import numpy as np
import operator
import warnings,os
import sys
import operator as opr
import qutilities as qutils
import utils as cutils
import operator,time
import functools as fct
import copy
try:
    snp=sys.modules['sparsenumpy']
except KeyError:
    import sparsenumpy as snp

herm=lambda x:np.conj(np.transpose(x))

def generate_binary_deferer(op_func):
    def deferer(cls, B, *args, **kwargs):
        return type(cls).__binary_operations__(cls, B, op_func, *args,**kwargs)
    return deferer

def generate_in_place_binary_deferer(op_func):
    def deferer(cls, B, *args, **kwargs):
        return type(cls).__in_place_binary_operations__(cls, B, op_func, *args,**kwargs)
    return deferer

def generate_unary_deferer(op_func):
    def deferer(cls, *args, **kwargs):
        return type(cls).__unary_operations__(cls, op_func, *args,**kwargs)
    return deferer

def generate_in_place_unary_deferer(op_func):
    def deferer(cls, *args, **kwargs):
        return type(cls).__in_place_unary_operations__(cls, op_func, *args,**kwargs)
    return deferer


def addColumns(df,*x):
    """
    takes a data frame df and columns x=[column1,column2,...,]
    and sums the columns of df
    returns a Series
    """
    x_=df[x[0]]
    for n in range(1,len(x)):
        x_+=df[x[n]]
    return x_


class BookKeeper:
    """
    BookKeeper class handles th ebook-keeping of the quantum number blocks
    Attributes:
    self._labels: a list of str labelling the tensor-legs
    self._indices: a list of int labeling the tensor legs by numbers
    """

    def __init__(self,keys,blocks,labels,rank):
        self.rank=rank
        self._blocks=copy.deepcopy(blocks)
        if np.all(labels!=None):
            self._labels=copy.deepcopy(labels)
            self._indices=list(range(rank))            
        else:
            self._labels=list(range(rank))
            self._indices=list(range(rank))

        if len(keys)>0:
            self.dtype=type(keys[0][0])                 
            self._data=np.empty((len(keys),self.rank),dtype=self.dtype)            
            for m in range(len(keys)):
                k=keys[m]
                if len(k)!=self.rank:
                    raise ValueError("BookKeeper.__init__(): one of the key-lengths is different from tensor rank")
                for n in range(len(k)):
                    if __debug__:
                        if type(k[n])!=self.dtype:
                            raise TypeError("in BookKeeper.__init__(): found type {0} for a key on leg {1}, which is different from previously found type {2}".format(type(k[n]),n,self._QN.dtype))
                    self._data[m][n]=copy.deepcopy(k[n])
        else:
            self.dtype=np.ndarray #default type
            self._data=np.empty((len(keys),self.rank),dtype=self.dtype)                        
        
        
    def update_hashes(self):
        if np.issubdtype(self.dtype,np.dtype(int)):
            self._hash=np.asarray([hash(np.array(self._data[n,:]).tostring()) for n in range(self._data.shape[0])])
        elif (self.dtype==np.ndarray) or (self.dtype==tuple):
            self._hash=np.asarray([hash(np.concatenate(self._data[n,:],axis=0).tostring()) for n in range(self._data.shape[0])])
        else:
            raise TypeError("BookKeeper.update_hashes(): unknown quantum number type {0}; use integer or ndarray.".format(self.dtype))
        return self        

    def drop_duplicates(self):
        """
        BookKeeper.drop_duplicates():
        drops all tensors with duplicate quantum numbers
        """
        try:
            unique,inds=np.unique(self._hash,return_index=True)
        except AttributeError:
            self.update__hashes()
            unique,inds=np.unique(self._hash,return_index=True)            
        self._data=self._data[inds,:]
        temp=[self._blocks[k] for k in inds]
        self._blocks=temp
        self.update_hashes()
        return self
        
    def __len__(self):
        return len(self._blocks)
    
    @property
    def blocks(self):
        return self._blocks
    
    @property
    def labels(self):
        """
        return a list of the str labels of the tensor legs
        """
        return self._labels
    @labels.setter
    def labels(self,labellist):
        self._labels=labellist
        return self

    @property
    def labelindices(self):
        """
        return a list containing the integer-labels of the tensor legs
        """
        return self._indices
    @labelindices.setter
    def labelindices(self,indexlist):
        self._indices=indexlist
    
    @property
    def DataFrame(self):
        return pd.DataFrame.from_records(data=self._data,columns=self.labels)
    
    def __str__(self):
        print('')
        print('BookKeeper')
        print(self.DataFrame)
        return ''
    
    def checkconsistency(self,verbose=0):    
        """ checks if the tensor blocks have consistent shapes; if shapes are inconsistent raises an error"""
        df=self.DataFrame
        try:
            for leg in range(self.rank):
                try:
                    for k,v in df.groupby(self.labels[leg],sort=False).groups.items():
                        Ds=list(map(lambda x: np.shape(x)[leg],[self._blocks[index] for index in v.tolist()]))
                        if not all(D==Ds[0] for D in Ds):
                            raise ValueError("SparseTensor.checkconsistency found inconsistent block shapes for leg {0} and quantum number {1}".format(self.labels[leg],k))
                except TypeError:
                    df=df.applymap(tuple)
                    for k,v in df.groupby(self.labels[leg],sort=False).groups.items():
                        Ds=list(map(lambda x: np.shape(x)[leg],[self._blocks[index] for index in v.tolist()]))
                        if not all(D==Ds[0] for D in Ds):
                            raise ValueError("SparseTensor.checkconsistency found inconsistent block shapes for leg {0} and quantum number {1}".format(leg,k))
            if df.shape[0]!=len(self._blocks):
                raise ValueError("SparseTensor.checkconsistency(): number of tensor blocks is different from the number of rows in the dataframe")
        except ValueError as error:
            raise ValueError
        else:
            if verbose>0:
                print('SparseTensor.checkconsistency: tensor is consistent')
        

class BookKeeperView(BookKeeper):
    def __init__(data,blocks,labels,rank):
        NotImplemented
        self.rank=self.rank
        self._data=data.view()
        self._blocks=[t.view() for t in blocks]
        self.labels=labels

    
"""
Tensor index is essentially a wrapper for a dict() and a flow; it stores all desired (quantum number,dimension) pairs on a tensor leg as (key,value) pairs. _flow is the flow direction of the leg
A list of TensorIndex objects can be used to initialize a SparseTensor (see below); any type which suports arithmetic operations (+,-,*) can be used
as a quantum number, with the exception of tuple()
"""
class TensorIndex(object):
    @classmethod
    def fromlist(cls,quantumnumbers,dimensions,flow,label=None):
        if all(map(np.isscalar,quantumnumbers)):
            QNs=list(quantumnumbers)            
        elif all(list(map(lambda x: not np.isscalar(x),quantumnumbers))):
            QNs=list(map(np.asarray,quantumnumbers))
        else:
            raise TypeError("TensorIndex.fromlist(cls,dictionary,flow,label=None): quantum numbers have inconsistent types")
        return cls(QNs,dimensions,flow,label)
    
    @classmethod
    def fromdict(cls,dictionary,flow,label=None):
        if all(map(np.isscalar,dictionary.keys())):
            QNs=list(dictionary.keys())            
        elif all(list(map(lambda x: not np.isscalar(x),dictionary.keys()))):
            QNs=list(map(np.asarray,dictionary.keys()))
        else:
            raise TypeError("TensorIndex.fromdict(cls,dictionary,flow,label=None): quantum numbers have inconsistent types")

        return cls(QNs,list(dictionary.values()),flow,label)

    def __init__(self,quantumnumbers,dimensions,flow,label=None):
        if __debug__:
            if len(quantumnumbers)!=len(dimensions):
                raise ValueError("TensorIndex.__init__: len(quantumnumbers)!=len(dimensions)")

        try:
            unique=dict(zip(quantumnumbers,dimensions))
        except TypeError:
            unique=dict(zip(map(tuple,quantumnumbers),dimensions))


        if __debug__:
            if len(unique)!=len(quantumnumbers):
                warnings.warn("in TensorIndex.__init__: found some duplicate quantum numbers; duplicates have been removed")

        if __debug__:
            try:
                mask=np.asarray(list(map(len,unique.keys())))==len(list(unique.keys())[0])
                if not all(mask):
                    raise ValueError("in TensorIndex.__init__: found quantum number keys of differing length {0}\n all quantum number have to have identical length".format(list(map(len,unique.keys()))))
            except TypeError:
                if not all(list(map(np.isscalar,unique.keys()))):
                    raise TypeError("in TensorIndex.__init__: found quantum number keys of mixed type. all quantum numbers have to be either integers or iterables")
        self._data=np.array(list(zip(map(np.asarray,unique.keys()),dimensions)),dtype=object)
            
        self._flow=flow
        self.label=label
        
    def __getitem__(self,n):
        return self._data[n[0],n[1]]

    def Q(self,n):
        return self._data[n,0]
    
    def D(self,n):
        return self._data[n,1]
    
    def __len__(self):
        return self._data.shape[0]

    def setflow(self,val):
        if val==0:
            raise ValueError("TensorIndex.flow: trying to set TensorIndex._flow to 0, use positive or negative integers only")
        self._flow=np.sign(val)        
        return self
    
    def rename(self,label):
        self.label=label
        return self
    
    @property
    def flow(self):
        return self._flow
    
    @flow.setter
    def flow(self,val):
        if val==0:
            raise ValueError("TensorIndex.flow: trying to set TensorIndex._flow to 0, use positive or negative integers only")
        self._flow=np.sign(val)        

    @property
    def shape(self):
        return self._data.shape
    
    @property
    def DataFrame(self):
        return pd.DataFrame.from_records(data=self._data,columns=['qn','D'])
    
    def __str__(self):
        print('')
        print('TensorIndex, label={0}, flow={1}'.format(self.label,self.flow))
        print(self.DataFrame)
        return ''
        
        

"""    
IMPORTANT NOTE
you can use any object as key for the SparseTensor class, EXCEPT TUPLES! If you want to use something with the same 
properties as a tuple, then use a class derived from tuple; the reason is that SparseTensor uses a classification system
of identifying a key as a (nested) tuple, to see wether the index is combined or not; if it is a (nested) tuple, 
Sparseit is assumed that it can be split again; Any key-class has to implement operator.add (addition), operator.neg such that 
for example (-1)*a+b is possible, where a and b are some keys

HOW TO INITIALIZE
to create a sparse tensor of rank R, you can call tensor=SparseTensor.empty(R,qflow), and then 
fill in key-value pairs using operator []:
for k,v in zip(keys,values):
    tensor[k]=v
this takes care of the correct initialization of all internal members. qflow is a tuple of 1 and -1 values, 
and len(qflow)=tensor._rank, which tells the code
which of the tensor legs are inflowing (1) and which are outflowing (-1). 
"""

class SparseTensor(object):

    #_df=pd.DataFrame()  #a static dataframe shared by all SparseTensor objects to facilitate book-keeping
    @classmethod
    def numpy_initializer(cls,fun,indices,dtype,conserve,ktoq=None,*args,**kwargs):
        """
        initializes the tensors with fun; indices are a list of TensorIndex objects carrying the relevant information 
        to initialize the tensor; 
        Parameters:
        ----------------------------------
        fun: function for initialization
             signature of fun isL fun(shape,*args,**kwargs), where shape is a tuple of integers
             could be np.zeros , np.random.random_sample, np.ones, ...
        indices:  list containing TensorIndex objects:
                  the indices of the tensor. indices[n] holds the quantum-number to bond-dimension mapping for leg n
        dtype:  numpy.dtype of {bool,int,float,complex}
                the dtype of the tensor elements
        conserve: bool
                  if True, function only keeps tensors which have a total charge of zero
                  if False, all possible blocks are constructed from indice; note that this can generate large tensors with many blocks
        """
        keys=[]
        shapes=[]
        qutils.getKeyShapePairs(n=0,indices=indices,keylist=keys,shapelist=shapes,key=[],shape=[],conserve=conserve)
        labels=[I.label for I in indices]
        uniquelabels=[]
        suffix={}
        for l in labels:
            if l==None:
                if len(uniqueintlabels)==0:
                    l=0
                else:
                    l=max(uniqueintlabels)+1
                uniqueintlabels.append(l)
                
            if l not in uniquelabels:
                uniquelabels.append(l)
            else:
                warnings.warn("numpy_initializer found duplicate labels for the tensor legs; renaming duplicates")                
                if isinstance(l,str):                
                    if l in suffix:
                        renamed=l+'_{0}'.format(suffix[l]+1)
                        suffix[l]+=1
                    else:
                        renamed=l+'_{0}'.format(1)
                        suffix[l]=1                                                
                    uniquelabels.append(renamed)                                        
                elif isinstance(l,int) or isinstance(l,np.int64)or isinstance(l,np.int32):
                    while l in uniquelabels:
                        l+=1
                    uniquelabels.append(l)
                else:
                    raise TypeError("in SparseTensor.numpy_initializer: unknown label type for tensor legs; allowed types are int or str")

        values=[fun(shape,*args,**kwargs).astype(dtype) for shape in shapes]                
        if __debug__:
            if len(keys)==0:
                warnings.warn("no keys provided for SparseTensor.numpy_initializer(cls,fun,indices,dtype,*args,**kwargs)",stacklevel=3)
            if len(keys)!=len(values):
                raise ValueError("in classmethod SparseTensor.numpy_initializer(cls,fun,indices,dtype,*args,**kwargs): len(keys)!=len(values)")
        qflow= np.asarray([I._flow for I in indices])
        #return cls(keys,values,qflow,dtype=dtype,keytoq=ktoq,defval=0,labels=labels)
        out=cls()
        if dtype==np.bool:
            out._defval=False
        elif np.issubdtype(dtype,np.dtype(int)):
            out._defval=np.int64(0)
        elif np.issubdtype(dtype,np.dtype(float)) :           
            out._defval=np.float64(0)            
        elif np.issubdtype(dtype,np.dtype(complex)) :                       
            out._defval=np.complex128(0)
            
        out._dtype=dtype
        out._QN=BookKeeper(keys=keys,blocks=values,labels=uniquelabels,rank=len(qflow))
        out._QN.update_hashes()
        if len(keys)>0:
            out.drop_duplicates()
        out.qflow= np.asarray([I._flow for I in indices])
        out._ktoq=ktoq
        return  out


    @classmethod
    def eye(cls,index,qflow=[1,-1],dtype=np.float64,*args,**kwargs):
        
        """
        Initialize a eye-matrix from a TensorIndex object index
        SparseTensor.eye(I) creates an rank-2 tensor with identities
        """
        i1=copy.deepcopy(index)

        i1._flow=qflow[0]
        index._flow=qflow[1]
        iden=cls.numpy_initializer(np.empty,[i1,index],dtype,conserve=True,*args,**kwargs)
        for n in range(len(iden)):
            iden.tensors[n]=np.eye(iden.tensors[n].shape[0])
            
        return iden

    
    @classmethod
    def random(cls,indices,dtype=np.float64,conserve=True,*args,**kwargs):
        """
        Initialize a random tensor from a list of TensorIndex objects:
        SparseTensor.random([I1,I2,...,IN]) creates an rank-N tensor 
        with block initialized according to I1,...,IN
        """
        
        return cls.numpy_initializer(np.random.random_sample,indices,dtype,conserve,*args,**kwargs)
    
    @classmethod
    def zeros(cls,indices,dtype=np.float64,conserve=True,*args,**kwargs):
        """
        Initialize a tensor of zeros from a list of TensorIndex objects
        SparseTensor.zeros([I1,I2,...,IN]) creates an rank-N tensor 
        with block initialized according to I1,...,IN
        """
        return cls.numpy_initializer(np.zeros,indices,dtype,conserve,*args,**kwargs)
   
    @classmethod
    def ones(cls,indices,dtype=np.float64,conserve=True,*args,**kwargs):
        """
        Initialize a tensor of ones from a list of TensorIndex objects
        SparseTensor.ones([I1,I2,...,IN]) creates an rank-N tensor 
        with block initialized according to I1,...,IN
        """
        return cls.numpy_initializer(np.ones,indices,dtype,conserve,*args,**kwargs)
    
    @classmethod
    def empty(cls,indices,dtype=np.float64,conserve=True,*args,**kwargs):
        """
        Initialize a tensor of empty form from a list of TensorIndex objects
        SparseTensor.empty([I1,I2,...,IN]) creates an rank-N tensor 
        with block of sizes according to I1,...,IN, but all uninitialized (see numpy.empty)
        """
        return cls.numpy_initializer(np.empty,indices,dtype,conserve,*args,**kwargs)
    
    @classmethod
    def tensor_initializer(cls,fun,tensor,*args,**kwargs):    
        """
        SparseTensor.tensor_initializer(fun,tensor,*args,**kwargs):    
        initialize a new tensor with identical properties as "tensor",  but blocks initialized with 
        fun
        """
        keys=[tensor.getQN(n) for n in range(len(tensor))]
        values=[fun(np.shape(t),*args,**kwargs).astype(tensor.dtype) for t in tensor.tensors]
        #return cls.numpy_initializer(np.random.random_sample,Indices,tensor.dtype,*args,**kwargs)
        return cls(keys,values,tensor.qflow,labels=tensor.labels,dtype=tensor.dtype)
    
    @classmethod    
    def random_like(cls,tensor,*args,**kwargs):
        """
        returns a new tensor with the same rank, qflow and dataframe as tensors, but
        tensor block initialized randomly
        """
        #return cls.numpy_initializer(np.random.random_sample,Indices,tensor.dtype,*args,**kwargs)
        return cls.tensor_initializer(np.random.random_sample,tensor)
    
    @classmethod
    def zeros_like(cls,tensor,*args,**kwargs):
        """
        returns a new tensor with the same rank, qflow and blokc-data as tensors, but
        tensor block initialized with zeros
        """

        #Indices=[tensor.Index(n) for n in range(tensor.rank)]
        #return cls.numpy_initializer(np.zeros,Indices,tensor.dtype,*args,**kwargs)
        return cls.tensor_initializer(np.zeros,tensor)        
    @classmethod
    def ones_like(cls,tensor,*args,**kwargs):
        """
        returns a new tensor with the same rank, qflow and blokc-data as tensors, but
        tensor block initialized with ones
        """

        #Indices=[tensor.Index(n) for n in range(tensor.rank)]
        #return cls.numpy_initializer(np.ones,Indices,tensor.dtype,*args,**kwargs)
        return cls.tensor_initializer(np.ones,tensor)        

    @classmethod
    def empty_like(cls,tensor,*args,**kwargs):
        """
        returns a new tensor with the same rank, qflow and blokc-data as tensors, but
        tensor block not initialized  (see numpy.empty)
        """
        #Indices=[tensor.Index(n) for n in range(tensor.rank)]
        #return cls.numpy_initializer(np.empty,Indices,tensor.dtype,*args,**kwargs)
        return cls.tensor_initializer(np.empty,tensor)        
    
    @classmethod
    def fromblocks(cls,keys,values,qflow,dtype=np.float64,labels=None):
        """
        @classmethod
        fromblocks(keys,values,qflow):
        Initializes a tensor from a list of keys and values with given qflow
        """
        if len(keys)==0:
            warnings.warn('in classmethod SparseTensor.fromblocks(keys,values): initializing empty tensor')
        if len(keys)!=len(values):
            raise ValueError("in classmethod SparseTensor.fromblocks(keys,values): len(keys)!=len(values)")
        return cls(keys,values,qflow,dtype=dtype,keytoq=None,defval=0.0,labels=labels)
        
    """
    keys:  
    list of tuples; the length of each tuple keys[n] is equal to the rank of the tensor; the elements of the tuple key[n] 
    are related to the quantum numbers, one for each leg of the tensor. For non-degenerate quantum numbers, key[n] can be the 
    quantum number itself. If the quantum numbers are degenerate, then the user can optionally 
    provide a dict() in  keytoq[n]=dict() which  maps the key[n] to a valid quantum number object such that it matches the 
    type of the quantum number objects of the other legs. the total number N=len(keys) is the number of non-zero blocks of the tensor.

    values: 
    a list of length N of mapped values, usually of type ndarray. If other types are given, they should implement member functions 
    so that they can be called with np.reshape and np.tensordot.

    qflow: 
    a tuple of length len(Ds) of numbers that defines flow-directions of the legs of the tensor. for a positive element 
    sign(qflow[n][0]) at index n, the quantum leg is inflowing, for a negative value sign(qflow[n]) it is outflowing. when the tensor 
    has combined indices, then qflow will contain nested tuples. The charge of a leg is determind from multiplying its qflow value
    with its charge value. E.g. for  qflow=((1,1),1), a key of ((3,4),-7) has charge 0, and for ((1,1),-1) a key ((3,4),-7) has 
    charge 14.

    keytoq: 
    a tuple of len(Ds) of either None or dict()-type. For any n with type(keytoq[n])==dict(), the quantum number q of a key k on leg n 
    is calculated form q=keytoq[n][k]. keytoq[n] has to map each possible key k on leg n onto a quantum number q such that q is of 
    the same type as the quantum numbers of any other leg m with keytoq[m]==None. This is convenient if quantum numbers on a leg are 
    degenerate, like for example for spin half electrons with only charge (or only spin) conservation. 
    One can then use a physical index with four values corresponding to (0,0),(1,0),(0,1),(1,1) 
    electrons of type (up,down). If only charge is conserved, then the virtual legs of the tensor carry a single number 
    corresponding to charge. The physical indices 1=(1,0) 
    and 2=(0,1) are degenerate in this case, since they both carry charge 1.  A possible dict() for this case is 
    d={0:0,1:1,2:1,3:2} if keytoq[n]==None, it is assumed that the keys of the leg are identical to the quantum numbers: k=q
    Note that the use of ktoq is unneccessary for DMRG applications, since one can always choose the MPS and MPO tensor 
    quantum numbers in such a way that such degeneracies are automatically taken care of. It is however convenient to have this 
    feature in order to use keys that do not directly implement __add__ and __sub__ methods (for example strings).

    dtype: 
    data type of the underlying ndarray: can be float or complex

    defval: 
    the default value of the tensor outside of the blocks (0.0 per default)

    """
    def __init__(self,*args,dtype=np.float64,keytoq=None,defval=0.0,labels=None):
        if len(args)>0:
            keys=args[0]
            values=args[1]
            qflow=args[2]
            if dtype==np.bool:
                self._defval=np.bool(defval)
            elif np.issubdtype(dtype,np.dtype(int)):
                self._defval=np.int64(defval)
            elif np.issubdtype(dtype,np.dtype(float)) :           
                self._defval=np.float64(defval)            
            elif np.issubdtype(dtype,np.dtype(complex)) :                       
                self._defval=np.complex128(defval)
            if __debug__:
                if len(keys)!=len(values):
                    raise ValueError("SparseTensor.__init__(): length of key-list is different from length of block-list")
                
            
            #the datatype of the tensor            
            self._dtype=dtype            
            #keytoq is a list that for each leg of the tensor contains either None or a dict(). If indextoq[n]=None, it means that
            #the quantum number of leg n is already the key. If its a dict(), the dict has to map the keys of leg n to a quantum number
            #such that it is identical to the quantum number types used on the other legs
            if keytoq==None:
                self._ktoq=tuple(list([None])*len(qflow))
            else:
                self._ktoq=tuple(keytoq)
            
            #the block-labels are stored in BookKeeper class in an ndarray
            #each column labels a different feature of the tensor:
            #e.g.: q1,q2,...,qN are quantum numbers for each leg 1,..,N
            #BookKeeper also hold the tensor blocks and labels of the legs
            self._QN=BookKeeper(keys=keys,blocks=values,labels=labels,rank=len(qflow))
            self._QN.update_hashes()
            if len(keys)>0:
                self.drop_duplicates()
            self.qflow=np.asarray(qflow)


            
    def Index(self,leg,from_labels=False):
        """
        Index(leg,from_labels=False):
        returns a TensorIndex object corresponding to leg "leg" of the SparseTensor
        leg (int, or element in SparseTensor.labels if from_labels=True): the leg for which the TensorIndex object should be constructed
        """
        
        quantumnumbers,dimensions=self.blockdims(leg,from_labels)
        if (from_labels==True):
            if(any([l==leg for l in self.labels])):
                index=np.nonzero(np.asarray(self.labels,dtype=object)==leg)[0][0]            
            else:
                raise ValueError("in SparseTensor.Index(leg,from_labels=False): label {0} not found in SparseTensor.labels".format(leg))

            return TensorIndex(quantumnumbers,dimensions,self.qflow[index],self.labels[index])                        
        else:
            if not np.issubdtype(type(leg),np.dtype(int)):
                raise TypeError("SparseTensor.Index(leg,from_labels=False): type(leg)={0} is not of integer type".format(type(leg)))
            if leg>=self.rank:
                raise ValueError("in SparseTensor.Index(leg,from_labels=False): leg>=self.rank")
            return TensorIndex(quantumnumbers,dimensions,self.qflow[leg],self.labels[leg])                        
        
    @property
    def tensors(self):
        return self._QN.blocks

    @tensors.setter
    def tensors(self,newlist):
        self._QN._blocks=newlist

    @property
    def labels(self):
        return self._QN.labels
    @labels.setter
    def labels(self,labellist):
        self._QN.labels=labellist
        return self


    @property
    def labelindices(self):
        return self._QN.labelindices
    
    @labelindices.setter
    def labelindices(self,indexlist):
        self._QN.labelindices=indexlist
        return self
    
    
    @property
    def defval(self):
        """
        return the default value of the SparseTensor
        """
        return self._defval
    @property
    def ktoq(self):
        return self._ktoq
    
    @property
    def DataFrame(self):
        """
        return a pandas dataframe containing the quantum number blocks of the mps
        """
        return pd.DataFrame.from_records(data=self._QN._data,columns=self._QN.labels)
    
    @property
    def dtype(self):
        return self._dtype
    @property
    def rank(self):
        return self._QN.rank
    
    def drop_duplicates(self):
        """
        SparseTensor.drop_duplicates():
        drops all tensors with duplicate quantum numbers
        """
        self._QN.drop_duplicates()
        #df=self.DataFrame
        #print(df)
        #df=df.applymap(tuple)
        
        #t1=time.time()
        #df.drop_duplicates(subset=df.columns,keep='last',inplace=True)
        #t2=time.time()
        #print('datafrme took ', t2-t1)

        #t1=time.time()
        #unique,inds=np.unique(self._QN._hash,return_index=True)
        #t2=time.time()
        #print('np.unique took ', t2-t1)

        #self._QN._data=self._QN._data[inds,:]
        #temp=[self._tensors[k] for k in inds]
        #self._tensors=temp
        #self._QN.update_hashes()
        
        if len(self.tensors)!=len(self._QN):
            raise ValueError("in SparseTensor.drop_duplicates(): len(self.tensors)={0} is different from len(self._QN)={1}: something went wrong here!".format(len(self.tensors),len(self._QN)))
        return self
    
    def view(self):
        """
        returns an independent object SparseTensor that shares data with self. i.e., the tensor-blocks
        of self and the returned object are the same things (the same stuff in memory). Modification of one 
        also modifies the other. The exception to that is the DataFrame object df, which might be copied (I'm not sure
        about it)
        """
        raise NotImplementedError
        view=SparseTensor([],[],self.qflow,dtype=self._dtype,keytoq=self._ktoq,defval=self._dfval)
        view._QN=self._QN.view()
        
    def flipflow(self):
        """
        flips the flow direction of all tensor legs in place;
        """
        self.qflow*=(-1)
        return self
    
    def flippedflow(self):
        """
        returns the flipped flow direction of the tensor
        """        
        return self.qflow*(-1)
    
    @property
    def shape(self):
        """
        return a tuple of the tensor-dimension (similar to ndarray.shape)
        """        
        return self._shape()

    def _shape(self):
        """
        return a tuple of the tensor-dimension (similar to ndarray.shape)
        """        
        return tuple([self.dim(n) for n in range(self.rank)])

    
    def checkconsistency(self,verbose=0):
        """ 
        checks if the tensor blocks have consistent shapes; if shapes are inconsistent raises a ValueError
        
        """
        df=self.DataFrame
        try:
            for leg in range(self.rank):
                print()                                
                print('leg ',leg)
                try:
                    for k,v in df.groupby(self.labels[leg],sort=False).groups.items():
                        Ds=list(map(lambda x: np.shape(x)[leg],[self.tensors[index] for index in v.tolist()]))
                        if not all(D==Ds[0] for D in Ds):
                            raise ValueError("SparseTensor.checkconsistency found inconsistent block shapes for leg {0} and quantum number {1}".format(self.labels[leg],k))
                except TypeError:
                    df=df.applymap(tuple)
                    for k,v in df.groupby(self.labels[leg],sort=False).groups.items():
                        Ds=list(map(lambda x: np.shape(x)[leg],[self.tensors[index] for index in v.tolist()]))
                        if not all(D==Ds[0] for D in Ds):
                            raise ValueError("SparseTensor.checkconsistency found inconsistent block shapes for leg {0} and quantum number {1}".format(leg,k))
            if df.shape[0]!=len(self.tensors):
                raise ValueError("SparseTensor.checkconsistency(): number of tensor blocks is different from the number of rows in the dataframe")
        except ValueError as error:
            raise ValueError
        else:
            if verbose>0:
                print('SparseTensor.checkconsistency: tensor is consistent')
        return self
    @property
    def keytype(self):
        return self._QN.dtype
    
    def dim(self,leg,from_labels=False):
        
        """
        SparseTensor.dim(leg):
        returns the total dimension of the tensor leg "leg"
        """
        return np.sum(self.blockdims(leg,from_labels)[1])

    def blockdims(self,leg,from_labels=False):
        """
        SparseTensor.dim(leg):
        returns: a tuple of two lists holding the quantum numbers and their block-dimensions
                 [q1,a2,..,qN],[d1,d2,...,dN]
                 such that quantum nuber q1 has dimension d1, and so on
        leg (int or label in self.labels): leg for which to return the lists
        from_labels (bool): if True, leg has to be a label name, if False; leg is the integer label of the tensor leg
        """
        if from_labels==False:
            if not np.issubdtype(type(leg),np.dtype(int)):
                raise TypeError("SparseTensor.blockdims(leg,from_labels=False): type(leg)={0} is not of integer type".format(type(leg)))
            try:
                return self.blockdimsFromInts(leg)
            except ValueError:
                raise ValueError("in SparseTensor.blockdims(leg): leg>self.rank ")

        else:
            try:
                return self.blockdimsFromLabels(leg)
            except ValueError:
                raise ValueError("in SparseTensor.blockdims(leg): label {0} not found in SparseTensor.labels".format(leg))                
        
    def blockdimsFromLabels(self,leg):
        """
        SparseTensor.dim(leg):
        returns: a tuple of two lists holding the quantum numbers and their block-dimensions
                 [q1,a2,..,qN],[d1,d2,...,dN]
                 such that quantum nuber q1 has dimension d1, and so on
        leg (label in self.labels): leg for which to return the lists
        """
        if any([l==leg for l in self.labels]):
            legind=np.nonzero(np.asarray(self.labels,dtype=object)==leg)[0][0]
            l=[hash(np.asarray(q).tostring()) for q in self._QN._data[:,legind]]
            unique,inds=np.unique(l,return_index=True)
            return [self._QN._data[n,legind] for n in inds],[np.shape(self.tensors[n])[legind] for n in inds]
        else:
            raise ValueError("in SparseTensor.blockdimsFromLabels(leg): label {0} not found in SparseTensor.labels".format(leg))
        
    def blockdimsFromInts(self,leg):
        """
        returns: a tuple of two lists holding the quantum numbers and their block-dimensions
                 [q1,a2,..,qN],[d1,d2,...,dN]
                 such that quantum nuber q1 has dimension d1, and so on
        leg (int): leg for which to return the lists
        """
        if leg>self.rank:
            raise ValueError("in SparseTensor.blockdimsFromInts(leg): leg>self.rank ")
        return self.blockdimsFromLabels(self.labels[leg])

    
    def charge(self,as_list=False):
        """
        SparseTensor.charge():
        returns a list of the total charge of each block, i.e. the sum of its individual quantum numbers multiplied by its qflow; 
        for a symmetric tensor, each charge has to be zero
        return type: list of self.keytype
        """

        QN=self._QN._data*self.qflow
        if not as_list:            
            return [sum(QN[p,:]) for p in range(len(self))]
        else:
            try:
                return list(map(list,[sum(QN[p,:]) for p in range(len(self))]))
            except TypeError:
                return list([sum(QN[p,:]) for p in range(len(self))])

    def blockshapes(self):
        """
        returns a list with the shape of each block
        """
        return list(map(np.shape,self.tensors))

    def getQN(self,n):
        """
        """
        #df=pd.DataFrame.from_records(data=self._QN._data,columns=self.labels)
        #return df.iloc[n,:].tolist()
        return self._QN._data[n,:]        

    def __len__(self):
        return len(self.tensors)
    
    def randomize(self,scaling=0.5):
        """
        randomizes the tensor in place
        """
        for k in range(len(self)):
            if np.issubdtype(self.dtype,np.dtype(float)):            
                self._QN._blocks[k]=(np.random.random_sample(self._QN._blocks[k].shape)-0.5)*scaling
            if np.issubdtype(self.dtype,np.dtype(complex)):                            
                self._QN._blocks[k]=(np.random.random_sample(self._QN._blocks[k].shape)-0.5+1j*(np.random.random_sample(self._QN._blocks[k].shape)-0.5))*scaling

    def insert(self,QN,block):
        """
        insert a quantum-number block pair (QN,block)
        function checks if duplicate exists, if yes, it overwrites it.
        QN can be either obtained from SparseTensor.getQN(n) (i.e. an np.ndarray of np.ndarrays of shape (rank,))
        or it can be a list of lists
        or it can be a list of np.ndarrays
        or it can be a list of integers
        """
        if(len(QN)!=self.rank):
            raise ValueError("SparseTensor.insert(QN,block): len(QN)!=self.rank")

        #    k=QN[n]
        #    if type(k)!=self._QN._data.dtype or not np.isscalar(k):
        #        raise TypeError("in SparseTensor.insert(QN,block): type {0} for QN-key on leg {1} is different from previously found type {2}".format(type(k),n,self._QN._data.dtype))
        #l=np.asarray([self._QN._data[n,:].tostring() for n in range(self._QN._data.shape[0])])
        #if not isinstance(type(QN[0]),type(self._QN._data[0][0])):
        #    raise TypeError("SparseTensor.inert(QN.block): type of QN-quantum numbers differs from the type in SparseTensor._QN")
        if isinstance(QN,list):
            if all(map(lambda x: isinstance(x,collections.Iterable),QN)):
                QN_=np.empty(len(QN),dtype=object)
                for n in range(len(QN)):                
                    QN_[n]=np.asarray(QN[n]) 
                inds=np.nonzero(self._QN._hash==hash(np.concatenate(QN_,axis=0).tostring()))[0]
            else:#if all(map(np.isscalar,QN)):
                QN_=np.asarray(QN)
                inds=np.nonzero(self._QN._hash==hash(QN_.tostring()))[0]
        elif isinstance(QN,np.ndarray):
            if all(map(np.isscalar,QN)):
                QN_=np.asarray(QN)
                inds=np.nonzero(self._QN._hash==hash(QN_.tostring()))[0]
            else:
                QN_=np.empty(len(QN),dtype=object)
                for n in range(len(QN)):                
                    QN_[n]=np.asarray(QN[n]) 
                inds=np.nonzero(self._QN._hash==hash(np.concatenate(QN_,axis=0).tostring()))[0]
                #QN_=QN
                #inds=np.nonzero(self._QN._hash==hash(np.concatenate(QN_,axis=0).tostring()))[0]
        else:
            raise TypeError("SparseTensor.insert(QN,block): QN has to be either of: 1) a list of lists, 2) a list of ints or 3) an np.ndarray of shape (rank,) holding np.ndarrays")

        if inds.size>0:
            self._QN._blocks[inds[0]]=block.astype(self.dtype)
        else:
            self._QN._data=np.append(self._QN._data,np.expand_dims(QN_,0),axis=0)
            self._QN._blocks.append(block.astype(self.dtype))

        self._QN.update_hashes()
        
    def __getitem__(self,ind):
        """
        SparseTensor__getitem__(ind):
        ind: integer;
        returns: ndarray; block at position ind
        """
        return self.tensors[ind]
    

    def __setitem__(self,QN,value):
        """
        insert a quantum-number block pair (QN,block)
        """
        self.insert(QN,value)
        return self


    def remove(self,n):
        """
        removes block "n" from the tensors and from the BookKeeper data
        n: int
           the index of the block to be removed
        """
        if n>len(self):
            raise IndexError("SparseTensor.remove(n): n is out of bounds")
        self._QN._blocks.pop(n)
        self._QN._data=self._QN._data[np.nonzero(np.arange(len(self))!=n),:]
        self._QN._hash=self._QN._hash[np.nonzero(np.arange(len(self))!=n)]

    def squeeze(self,thresh=1E-14):
        """
        removes all blocks with a total norm smaller then thresh
        """
        inds=np.nonzero(np.fromiter(map(np.linalg.norm,self.tensors),dtype=np.float64)>thresh)[0]
        self._QN._data=self._QN._data[inds,:]
        temp=[self.tensors[k] for k in inds]
        self.tensors=temp
        self._QN.update_hashes()
        return self
    

    def removeNonConserved(self):
        """
        removes all blocks with non-zero charge
        """
        charges=self.charge()
        zero=charges[0]*0
        inds=np.nonzero(list(map(lambda x: np.all(operator.__eq__(zero,x)), charges)))[0]
        self._QN._data=self._QN._data[inds,:]
        temp=[self.tensors[k] for k in inds]
        self.tensors=temp
        self._QN.update_hashes()
        return self
        
    #for use with print command
    def __str__(self):
        print('SparseTensor of length ',len(self.tensors))
        for k in range(len(self)):
            print ()
            print ("###################")
            print ()            
            print ("Quantum Numbers: {0}, shape={1}".format(list(self._QN._data[k,:]),self.tensors[k].shape))
            print ()
            print ("Block")            
            print (self.tensors[k])
        return ''

    def blockmax(self):
        """ 
        blockmax():
        returns a list of containing the maximum value of each block
        """
        return list(map(np.max,self.tensors))
    
    def blocknorms(self):
        """ 
        blocknorms():
        returns a list of containing the norms of each block
        """
        return list(map(np.linalg.norm,self.tensors))
    
    def norm(self):
        """ 
        norm():
        return the L2 norm of the tensor
        """
        return np.sqrt(sum(map(lambda x: x**2, self.blocknorms())))
        
    def normalize(self):
        """ 
        normalize():
        normalize the tensor according to the L2 norm
        """
        Z=self.norm()
        for T  in self.tensors:
            T/=Z

    def __in_place_unary_operations__(self,operation,*args,**kwargs):
        
        """
        __in_place_unary_operations__(operation,*args,**kwargs):
        act on all blocks with "operation" in place
        """
        self._defval=operation(self._defval,*args,**kwargs)
        if (self._defval!=0.0) and (self._defval !=False):
            warnings.warn('SparseTensor.__unary_operations__(operation,*args,**kwargs) resulted in non-invariant tensor',stacklevel=3)
        self._QN._blocks=list(map(lambda x: operation(x,*args,**kwargs),self.tensors))


    def __unary_operations__(self,operation,*args,**kwargs):
        """
        __unary_operations__(operation,*args,**kwargs):
        act on all blocks with "operation"
        """
        res=SparseTensor.empty_like(self)
        res._defval=operation(self._defval,*args,**kwargs)
        if (res._defval!=0.0) and (res._defval !=False):
            warnings.warn('SparseTensor.__unary_operations__(operation,*args,**kwargs) resulted in non-invariant tensor',stacklevel=3)
        res._QN._blocks=list(map(lambda x: operation(x,*args,**kwargs),self.tensors))
        return res
    
    def __binary_operations__(self,other,operation,*args,**kwargs):

        """
        __binary_operations__(other,operation,*args,**kwargs):
        act with "operation"(self[n],other) on all blocks 
        binary operations include multiplication with scalars, so other could be
        a scalar or a tensor. This functions is adapted from Markus Haurus abeliantensor.py
        code
        """
        #checks:
        if self.rank!=other.rank:
            raise ValueError("__binary_operations__(other,operation,*args,**kwargs): self and other have different ranks; cannot perform operation")
        res=SparseTensor([],[],self.qflow,dtype=self._dtype,keytoq=self._ktoq,defval=self._defval)        
        try:
            res._dtype=np.result_type(other.dtype,self.dtype)
        except AttributeError:
            res._dtype=np.result_type(other,self.dtype)

        if isinstance(other,SparseTensor):
            if np.all(self.qflow!=other.qflow):
                warnings.warn('SparseTensor.__binary_operations__(other,operation,*args,**kwargs): tensors have different qflow; using qflow of self',stacklevel=3)
            #if self and other have identical _QN, we can just go ahead and apply operation
            #on each block without any checks
            if np.all(self._QN._data==other._QN._data):
                res._defval=operation(self._defval,other._defval,*args,**kwargs)
                res._QN._data=np.copy(self._QN._data)
                res._QN._hash=np.copy(self._QN._hash)
                res._QN.rank=self._QN.rank
                res._QN._labels=np.copy(self._QN.labels)
                res._QN.dtype=self._QN.dtype
                res._QN._blocks=list(map(fct.partial(operation,*args,**kwargs),self._QN._blocks,other.tensors))
                #res._QN.update_hashes()
                return res
            
            #if self and other have different _QN, we need to get the common keys, the keys only present
            #in the left tensor and those only present in the right tensor
            else:
                common=np.intersect1d(self._QN._hash,other._QN._hash)

                commonindself=[np.nonzero(self._QN._hash==c)[0][0] for c in common]
                commonindother=[np.nonzero(other._QN._hash==c)[0][0] for c in common]

                self_only=np.setdiff1d(np.arange(len(self)), commonindself, assume_unique=True)
                other_only=np.setdiff1d(np.arange(len(other)), commonindother, assume_unique=True)

                tc=[operation(self[x],other[y],*args,**kwargs) for x,y in zip(commonindself,commonindother)]
                tl=[operation(self[x],other._defval,*args,**kwargs) for x in self_only]
                tr=[operation(self._defval,other[y],*args,**kwargs) for y in other_only]
                res._QN._data=np.concatenate((self._QN._data[commonindself,:],self._QN._data[self_only,:],other._QN._data[other_only,:]),axis=0)
                res._QN._hash=np.concatenate((self._QN._hash[commonindself],self._QN._hash[self_only],other._QN._hash[other_only]),axis=0)
                res._QN.rank=self.rank
                res._QN._labels=self._QN.labels
                res._QN.dtype=self._QN.dtype
                res._QN._blocks=tc+tl+tr
                return res
        else:
            res._defval = operation(self._defval,other,*args, **kwargs)
            res._QN._blocks=list(map(fct.partial(operation,*args,**kwargs),self.tensors,other))
            res._QN=copy.deepcopy(self._QN)
            if (res._defval!=0.0) and (res._defval !=False):
                warnings.warn('SparseTensor.__binary_operations__(self,operation,*args,**kwargs) resulted in non-invariant tensor',stacklevel=3)
            return res


    def __in_place_binary_operations__(self,other,operation,*args,**kwargs):

        """
        __in_place_binary_operations__(other,operation,*args,**kwargs):
        act with "operation"(self[n],other) on all blocks  in place
        binary operations include multiplication with scalars, so other could be
        a scalar or a tensor. This functions is adapted from Markus Haurus abeliantensor.py
        code
        """
        #checks:
        if self.rank!=other.rank:
            raise ValueError("__binary_operations__(other,operation,*args,**kwargs): self and other have different ranks; cannot perform operation")
        try:
            self._dtype=np.result_type(other.dtype,self.dtype)
        except AttributeError:
            self._dtype=np.result_type(other,self.dtype)

        if isinstance(other,SparseTensor):
            if np.all(self.qflow!=other.qflow):
                warnings.warn('SparseTensor.__binary_operations__(other,operation,*args,**kwargs): tensors have different qflow; using qflow of self',stacklevel=3)
            #if self and other have identical _QN, we can just go ahead and apply operation
            #on each block without any checks
            if np.all(self._QN._data==other._QN._data):
                self._defval=operation(self._defval,other._defval,*args,**kwargs)
                self._QN._data=np.copy(self._QN._data)
                self._QN._hash=np.copy(self._QN._hash)
                self._QN.rank=self._QN.rank
                self._QN._labels=np.copy(self._QN.labels)
                self._QN.dtype=self._QN.dtype
                self._QN._blocks=list(map(fct.partial(operation,*args,**kwargs),self._QN._blocks,other.tensors))
                #self._QN.update_hashes()                
                return self
            
            #if self and other have different _QN, we need to get the common keys, the keys only present
            #in the left tensor and those only present in the right tensor
            else:
                common=np.intersect1d(self._QN._hash,other._QN._hash)
                commonindself=[np.nonzero(self._QN._hash==c)[0][0] for c in common]
                commonindother=[np.nonzero(other._QN._hash==c)[0][0] for c in common]

                self_only=np.setdiff1d(np.arange(len(self)), commonindself, assume_unique=True)
                other_only=np.setdiff1d(np.arange(len(other)), commonindother, assume_unique=True)

                tc=[operation(self[x],other[y],*args,**kwargs) for x,y in zip(commonindself,commonindother)]
                tl=[operation(self[x],other._defval,*args,**kwargs) for x in self_only]
                tr=[operation(self._defval,other[y],*args,**kwargs) for y in other_only]
                self._QN._data=np.concatenate((self._QN._data[commonindself,:],self._QN._data[self_only,:],other._QN._data[other_only,:]),axis=0)
                self._QN._hash=np.concatenate((self._QN._hash[commonindself],self._QN._hash[self_only],other._QN._hash[other_only]),axis=0)
                self._QN.rank=self.rank
                self._QN._labels=self._QN.labels
                self._QN.dtype=self._QN.dtype
                self._QN._blocks=tc+tl+tr
                return self
        else:
            self._defval = operation(self._defval,other,*args, **kwargs)
            self._QN._blocks=list(map(fct.partial(operation,*args,**kwargs),self.tensors,other))            
            self._QN=copy.deepcopy(self._QN)
            if (self._defval!=0.0) and (self._defval !=False):
                warnings.warn('SparseTensor.__binary_operations__(self,operation,*args,**kwargs) resulted in non-invariant tensor',stacklevel=3)
            return res

    
    #from Markus Hauru's abeliantensor.py class
    def arg_swapper(op):
        def res(a,b, *args, **kwargs):
            return op(b,a, *args, **kwargs)
        return res

    __add__ = generate_binary_deferer(opr.add)
    __iadd__ = generate_in_place_binary_deferer(opr.iadd)
    __sub__ = generate_binary_deferer(opr.sub)
    __isub__ = generate_in_place_binary_deferer(opr.isub)    
    __mul__ = generate_binary_deferer(opr.mul)
    __imul__ = generate_in_place_binary_deferer(opr.imul)
    __rmul__ = generate_binary_deferer(arg_swapper(opr.mul))
    __truediv__ = generate_binary_deferer(opr.truediv)
    __itruediv__ = generate_in_place_binary_deferer(opr.itruediv)
    __floordiv__ = generate_binary_deferer(opr.floordiv)
    __mod__ = generate_binary_deferer(opr.mod)
    __divmod__ = generate_binary_deferer(divmod)
    __pow__ = generate_binary_deferer(pow)
    __lshift__ = generate_binary_deferer(opr.lshift)
    __rshift__ = generate_binary_deferer(opr.rshift)
    __and__ = generate_binary_deferer(opr.and_)
    __xor__ = generate_binary_deferer(opr.xor)
    __or__ = generate_binary_deferer(opr.or_)
    __radd__ = generate_binary_deferer(arg_swapper(opr.add))
    __rsub__ = generate_binary_deferer(arg_swapper(opr.sub))

    __rtruediv__ = generate_binary_deferer(arg_swapper(opr.truediv))
    __rfloordiv__ = generate_binary_deferer(arg_swapper(opr.floordiv))
    __rmod__ = generate_binary_deferer(arg_swapper(opr.mod))
    __rdivmod__ = generate_binary_deferer(arg_swapper(divmod))
    __rpow__ = generate_binary_deferer(arg_swapper(pow))
    __rlshift__ = generate_binary_deferer(arg_swapper(opr.lshift))
    __rrshift__ = generate_binary_deferer(arg_swapper(opr.rshift))
    __rand__ = generate_binary_deferer(arg_swapper(opr.and_))
    __rxor__ = generate_binary_deferer(arg_swapper(opr.xor))
    __ror__ = generate_binary_deferer(arg_swapper(opr.or_))

    __eq__ = generate_binary_deferer(opr.eq)
    __ne__ = generate_binary_deferer(opr.ne)
    __lt__ = generate_binary_deferer(opr.lt)
    __le__ = generate_binary_deferer(opr.le)
    __gt__ = generate_binary_deferer(opr.gt)
    __ge__ = generate_binary_deferer(opr.ge)

    __neg__ = generate_unary_deferer(opr.neg)
    __pos__ = generate_unary_deferer(opr.pos)
    __abs__ = generate_unary_deferer(abs)
    __invert__ = generate_unary_deferer(opr.invert)
    
    def __div__(self, other):   # Python 2 compatibility
        return type(self).__truediv__(self, other)
    def __idiv__(self, other):   # Python 2 compatibility
        return type(self).__itruediv__(self, other)
    

    def copy(self):
        """
        return a copy of self
        """
        return copy.deepcopy(self)


    def conjugate(self,*args,**kwargs):
        """
        return the conjugate of self
        upon conjugation, the flow-sign of the tensor is changed to minus its flowsign;
        this is neccessary when using SparseTensor for DMRG
        """
        
        res=self.__unary_operations__(np.conj,*args,**kwargs)
        res.qflow*=(-1)
        return res
    
    def conj(self,*args,**kwargs):
        """
        in-place conjugation of self
        upon conjugation, the flow-sign of the tensor is changed to minus its flowsign;
        this is neccessary when using SparseTensor for DMRG
        """
        
        self.__in_place_unary_operations__(np.conj,*args,**kwargs)
        self.qflow*=(-1)
        return self
    #take elementwise exp
    def exp(self,*args,**kwargs):
        res=self.__unary_operations__(np.exp,*args,**kwargs)
        return res
        
    #take elementwise sqrt
    def sqrt(self,*args,**kwargs):
        res=self.__unary_operations__(np.sqrt,*args,**kwargs)
        return res

    def real(self,*args,**kwargs):
        res=self.__unary_operations__(np.real,*args,**kwargs)
        res._dtype=np.dtype(float)
        return res

    def imag(self,*args,**kwargs):
        res=self.__unary_operations__(np.imag,*args,**kwargs)
        res._dtype=np.dtype(float)
        return res


    def T(self,*args):
        """
        def T(self,*args):
        in-place transpose the tensor
        args: empty, or a single list of the newindices
              SparseTensor.T() reverses the index order
              SparseTensor.T(newinds) sets index order to newinds
        """

        if len(args)>0:
            newinds=args[0]
            if np.all(np.sort(newinds)!=np.arange(self.rank)):
                raise ValueError("SparseTensor.transpose(newinds): newinds has to be a permutation of {0}; got {1}".format(np.arange(self.rank)),newinds)
        else:
            newinds=list(range(self.rank))[::-1]
        self.labels=[self.labels[n] for n in newinds]
        self.labelindices=[self.labelindices[n] for n in newinds]        
        for n in range(len(self)):
            self.tensors[n]=np.transpose(self.tensors[n],newinds)
        return self


    def transpose(self,*args):
        """
        def T(self,*args):
        returns the transpose the tensor
        args: empty, or a single list of the newindices
              SparseTensor.T() reverses the index order
              SparseTensor.T(newinds) sets index order to newinds
        """

        if len(args)>0:
            newinds=args[0]
            if np.all(np.sort(newinds)!=np.arange(self.rank)):
                raise ValueError("SparseTensor.transpose(newinds): newinds has to be a permutation of {0}; got {1}".format(np.arange(self.rank)),newinds)
        else:
            newinds=list(range(self.rank))[::-1]

        out=self.copy()
        out.labels=[out.labels[n] for n in newinds]
        out.labelindices=[out.labelindices[n] for n in newinds]        
        for n in range(len(out)):
            out.tensors[n]=np.transpose(out.tensors[n],newinds)
        return out
    
    def trace(self):
        if (self._defval!=0.0) or (self._defval!=False):
            warnings.warn("SparseTensor.trace(): defval of tensor!=0.0; the defval is not included in the trace ")
        tr=0.0
        inds=np.nonzero([np.all(list(map(lambda x: np.all(x==self._QN._data[n,0]),self._QN._data[n,:]))) for n in range(len(self))])[0]
        for i in inds:
            T=self.tensors[i]
            Dmin=np.min(T.shape)
            for s in range(Dmin):
                tr+=T[tuple([s]*self.rank)]
        return tr

    def herm(self,*args):
        """
        in-place hermitian conjuate of self
        """
        self.conj().T(*args)

    def hermitian(self,*args):
        """
        returns the hermitian conjuate of self
        """
        return self.copy().conj().T(*args)

                
    def todense(self):
        mapper=[dict() for s in range(self.rank)]
        for r in range(self.rank):
            Qs,Ds=self.blockdimsFromInts(r)
            start=0
            for n in range(len(Ds)):
                q=Qs[n]
                try:
                    mapper[r][tuple(q)]=slice(start,start+Ds[n],1)
                except TypeError:
                    mapper[r][q]=slice(start,start+Ds[n],1)
                start+=Ds[n]
        dense=np.zeros(self.shape,dtype=self.dtype)
        for n in range(len(self)):
            try:            
                slices=tuple([mapper[r][tuple(self._QN._data[n,r])] for r in range(self.rank)])
                dense[slices]=self.tensors[n]                
            except TypeError:
                slices=tuple([mapper[r][self._QN._data[n,r]] for r in range(self.rank)])
                dense[slices]=self.tensors[n]                
        return dense

    
    def fromdense(self,array):
        if np.any(np.asarray(array.shape)<np.asarray(self.shape)):
            raise ValueError("SparseTensor.fromdense(array): array.shape is too small to enclose self.shape are not matching")
        mapper=[dict() for s in range(self.rank)]
        for r in range(self.rank):
            Qs,Ds=self.blockdimsFromInts(r)
            start=0
            for n in range(len(Ds)):
                q=Qs[n]
                try:
                    mapper[r][tuple(q)]=slice(start,start+Ds[n],1)
                except TypeError:
                    mapper[r][q]=slice(start,start+Ds[n],1)
                start+=Ds[n]
        dense=np.zeros(self.shape,dtype=self.dtype)
        for n in range(len(self)):
            try:            
                slices=tuple([mapper[r][tuple(self._QN._data[n,r])] for r in range(self.rank)])
                self.tensors[n]=dense[slices]
            except TypeError:
                slices=tuple([mapper[r][self._QN._data[n,r]] for r in range(self.rank)])
                self.tensors[n]=dense[slices]
        return dense


