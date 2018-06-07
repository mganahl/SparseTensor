#!/usr/bin/env python
#from __future__ import division 
import numpy as np
import warnings,os
import sys
import operator as opr
import qutilities as utils
import operator
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

        
"""
Tensor index is essentially a wrapper for a dict() and a flow; it stores all desired (quantum number,dimension) pairs on a tensor leg as (key,value) pairs. _flow is the flow direction of the leg
A list of TensorIndex objects can be used to initialize a SparseTensor (see below); any type which suports arithmetic operations (+,-,*) can be used
as a quantum number, with the exception of tuple()
"""
class TensorIndex(object):
    @classmethod
    def fromlist(cls,Qs,Dims,flow):
        d=dict()
        d.update(zip(Qs,Dims))
        return cls(d,flow)
        
    def __init__(self,dictionary,flow):
        self._Q=dict()
        self._flow=flow
        self._Q.update(dictionary)
            

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
    """
    initializes the tensors with fun; indices are a list of TensorIndec objects carrying the relevant information 
    to initialize the tensor
    """
    @classmethod
    def numpy_initializer(cls,fun,indices,dtype,*args,**kwargs):
        keys=[]
        shapes=[]
        utils.getKeyShapePairs(n=0,indices=indices,keylist=keys,shapelist=shapes,key=[],shape=[])
        
        #keylist and shapelist contain the key and shape tuples allowed by the symmetry
        if len(keys)==0:
            warnings.warn("no keys provided for SparseTensor.numpy_initializer(cls,fun,indices,dtype,*args,**kwargs)",stacklevel=3)
        values=[]
        for shape in shapes:
            values.append(fun(shape,*args,**kwargs).astype(dtype))
        qflow=tuple([])
        for I in indices:
            qflow+=tuple([I._flow])

        return cls.fromblocks(keys=keys,values=values,qflow=qflow)

    @classmethod
    def tensor_initializer(cls,tensor,fun,*args,**kwargs):
        cls=tensor.__copy__()

        for k,v in cls._tensor.items():
            cls[k]=fun(v.shape,*args,**kwargs).astype(cls._dtype)
        return cls
    
    """
    Initialize a random tensor from a list of TensorIndex objects:
    SparseTensor.random([I1,I2,...,IN]) creates an rank-N tensor 
    with block initialized according to I1,...,IN
    """
    @classmethod
    def random(cls,indices,dtype=float,*args,**kwargs):
        return cls.numpy_initializer(np.random.random_sample,indices,dtype,*args,**kwargs)
    """
    Initialize a tensor of zeros from a list of TensorIndex objects
    SparseTensor.zeros([I1,I2,...,IN]) creates an rank-N tensor 
    with block initialized according to I1,...,IN

    """
    @classmethod
    def zeros(cls,indices,dtype=float,*args,**kwargs):
        return cls.numpy_initializer(np.zeros,indices,dtype,*args,**kwargs)        
    """
    Initialize a tensor of ones from a list of TensorIndex objects
    SparseTensor.ones([I1,I2,...,IN]) creates an rank-N tensor 
    with block initialized according to I1,...,IN
    """
    @classmethod
    def ones(cls,indices,dtype=float,*args,**kwargs):
        return cls.numpy_initializer(np.ones,indices,dtype,*args,**kwargs)        
    """
    Initialize a tensor of empty from a list of TensorIndex objects
    SparseTensor.empty([I1,I2,...,IN]) creates an rank-N tensor 
    with block initialized according to I1,...,IN
    """
    @classmethod
    def empty(cls,indices,dtype=float,*args,**kwargs):
        return cls.numpy_initializer(np.empty,indices,dtype,*args,**kwargs)        

    @classmethod
    def random_like(cls,tensor,*args,**kwargs):
        return cls.tensor_initializer(tensor,np.random.random_sample,*args,**kwargs)

    @classmethod
    def zeros_like(cls,tensor,*args,**kwargs):
        return cls.tensor_initializer(tensor,np.zeros,*args,**kwargs) 

    @classmethod
    def ones_like(cls,tensor,*args,**kwargs):
        return cls.tensor_initializer(tensor,np.ones,*args,**kwargs) 

    @classmethod
    def empty_like(cls,tensor,*args,**kwargs):
        return cls.tensor_initializer(tensor,np.empty,*args,**kwargs) 

    
    """
    Initializes a tensor from a list of keys and values; see below for explanation of qflow
    """
    @classmethod
    def fromblocks(cls,keys,values,qflow):
        if len(values)>0:
            if values[0].dtype==np.complex128:
                dtype=complex
            elif values[0].dtype==np.float64:
                dtype=float
            else:
                sys.exit("in SparseTensor.fromblocks: unknown value type")
                
        if len(keys)==0:
            sys.exit("in classmethod SparseTensor.fromblocks(keys,values): len(keys)=0; please enter list of length>0")
        if len(keys)!=len(values):
            sys.exit("in classmethod SparseTensor.fromblocks(keys,values): len(keys)!=len(values)")
        return cls(keys=keys,values=values,Ds=[dict() for n in range(len(keys[0]))],qflow=qflow,mergelevel=None,dtype=dtype,keytoq=None,defval=0.0)

    """
    Initializes an empty tensor; see below for explanation of qflow
    """
    @classmethod
    def empty(cls,rank,qflow,dtype=float):
        return cls(keys=[],values=[],Ds=[dict() for n in range(rank)],qflow=qflow,mergelevel=None,dtype=dtype,keytoq=None,defval=0.0)
        
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

    Ds: 
    a list of length N of dict()'s. Each dict() Ds[n] maps a set of keys of leg n into the dimension of the corresponding block.

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

    mergelevel: 
    tuple of len(Ds) containing strings 'A'. If the tensorlegs have been merged, mergelevel is a nested tuple 
    reflecting the way the tensors were merged. if mergelevel=None, a tensor with a trivial mergestructure is created.

    dtype: 
    data type of the underlying ndarray: can be float or complex

    defval: 
    the default value of the tensor outside of the blocks (0.0 per default)


    self._tensor:
    dict() mapping charge quantum numbers to ndarrays.
    The charge quantum numbers are (nested) tuples of whatever the user provides.

    self._shapes:
    a dict() mapping the different U(1) charge sectors of the tensor to the shapes of the tensors;
    shapes maps nested tuples of U(1) charge values to nested tuples of integers, corresponding to the shape 
    of the tensors. For example, a key-value pair ((1,2),3), ((10,11),200) corresponds to a matrix of shape
    (110,200) with U(1) charge 3 on one leg and 3 on the other. if qflow=((1,1),-1) the total charge is in this case 0).


    self._keys:
    list() of length self._rank (tensor-rank) of dict(); each dict() in self._keys[n] is a map which maps charge-keys
    on leg n to a list of tensor-keys with this specific value of charge on leg n. For example, self._keys[2] 
    is a dictionary of the form 
    {
       ((1,4),4):[(1,3,((1,4),4),5,9),(3,5,((1,4),4),5,6),...], 
       ((3,8),0):[(1,9,((3,8),0),9,9),(15,4,((3,8),0),59,1),...]
       .
       .
       .
    }
    This is convenient for quickly finding all blocks that have a given quantum number on a given leg.
    """
    
    def __init__(self,keys,values,Ds,qflow,mergelevel=None,dtype=float,keytoq=None,defval=0.0):
        self._defval=dtype(defval)
        self._Ds=[dict() for n in range(len(Ds))]
        self._rank=len(self._Ds)
        #keytoq is a list that for each leg of the tensor contains either None or a dict(). If indextoq[n]=None, it means that
        #the quantum number of leg n is already the key. If its a dict(), the dict has to map the keys of leg n to a quantum number
        #such that it is identical to the quantum number types used on the other legs
        if keytoq==None:
            self._ktoq=tuple(list([None])*self._rank)
        else:
            self._ktoq=tuple(keytoq)

        for n in range(len(Ds)):
            self._Ds[n]=copy.deepcopy(Ds[n])
        #the datatype of the tensor
        self._dtype=dtype
        #the blocks are stored in a dictionary (hash-table)
        #on initialization, create a list of dictionaries for each index; the dict maps each existing QN of the index to a list of tensor-keys, i.e. given a key k=0 in index n,
        self._tensor=dict()

        #self._key[n][k]=list() of all tensor-keys with with a value of k on the n-th index; 
        #this is useful for e.g. summations over a given index
        self._keys=[dict() for n in range(self._rank)]

        #this gives information about how nested the merged indices are. 0 means not nested at all (unmerged).
        if mergelevel==None:
            #self._mergelevel=tuple(range(self._rank))
            self._mergelevel=tuple(list('A')*self._rank)
        else:
            self._mergelevel=mergelevel

        #self._shapes is a dictionary that will contain the merged shape (i.e. a nested tuple()) of a block with a certain key
        self._shapes=dict()
        #insert values into the dict        
        for n in range(len(keys)):
            self[keys[n]]=values[n]

        self._qflow=qflow

        if list(self.__charge__().values())!=[0]*len(self.__keys__()):
            sys.exit('SparseTensor.__init__(): the tensor does not obey U(1) conservation of charge')
        #self.__shapes__() stores shapes in self._shapes
        self._shapes=self.__shapes__()


    """
    returns a copy of SparseTensor with an empty ._tensor dict(), that is other wise identical to self
    """
    def __empty__():
        return SparseTensor(keys=[],values=[],Ds=self._Ds,qflow=self._qflow,mergelevel=self._mergelevel,dtype=self._dtype)

    """
    returns an independent object SparseTensor that shares all dict() data with self. i.e., the tensor-blocks
    of self and the returned object are the same things (the same stuff in memory). Modification of one 
    also modifies the other.
    """
    def __view__(self):
        view=SparseTensor(keys=[],values=[],Ds=self._Ds,qflow=self._qflow,mergelevel=self._mergelevel,dtype=self._dtype)
        view._Ds=self._Ds.copy()
        view._keys=self._keys.copy()
        view._tensor=self._tensor.copy()
        view._ktoq=self._ktoq
        view._shapes=self._shapes.copy()

        
    """flips the flow direction of all tensor legs"""
    def __flipflow__(self):
        self._qflow=utils.flipsigns(self._qflow)
        return self

    """flips the flow direction of all tensor legs"""
    def __flippedflow__(self):
        return utils.flipsigns(self._qflow)
    
    
    """return a tuple of the tensor-dimension (similar to ndarray.shape)"""
    def __dims__(self):
        d=[]
        for n in range(self._rank):
            d.append(self.__dim__(n))
        return tuple(d)

    """returns the dimension of leg index"""
    def __dim__(self,index):
        dim=0
        for val in self._Ds[index].values():
            dim+=utils.prod(utils.flatten(val))
        return dim
    
    """returns a dict() of the total charge of each block; for a symmetric tensor, each charge should be zero"""
    def __charge__(self):
        charge=dict()
        flow=utils.flatten(self._qflow)
        for key in self.__keys__():
            #kflat=utils.tupsignmult(utils.flatten(key),utils.flatten(self._qflow))            
            #q=sum(kflat[1::],kflat[0])            
            
            k=utils.flatten(key)
            if self._ktoq[0]==None:
                q=k[0]*np.sign(flow[0])                
            else:
                q=self._ktoq[0][k[0]]*np.sign(flow[0])                
            for n in range(1,len(k)):
                if self._ktoq[n]==None:
                    q+=k[n]*np.sign(flow[n])
                else:
                    q+=self._ktoq[n][k[n]]*np.sign(flow[n])                    
            charge[key]=q
        return charge
    
    #returns a dict() with the shape of each block
    def __shapes__(self):
        shapes=dict()
        for k in self.__keys__():
            shape=[]
            for n in range(self._rank):
                shape.append(self._Ds[n][k[n]])
            shapes[k]=shape
        self._shapes=shapes
        return shapes

    #randomizes the tensor
    def __randomize__(self,scaling=0.5):
        for k in self.__keys__():
            shape=self[k].shape
            if self._dtype==float:
                self._tensor[k]=(np.random.random_sample(shape)-0.5)*scaling
            elif self._dtype==complex:
                self._tensor[k]=(np.random.random_sample(shape)-0.5+1j*(np.random.random_sample(shape)-0.5))*scaling
        return self
    
    #sets all values to 0
    def __zero__(self):
        for k in self.__keys__():
            shape=self[k].shape
            self._tensor[k]=np.zeros(shape).astype(self._dtype)
        return self
    
    #adds a single key to self._keys
    def __addkey__(self,key):
        try:
            assert(len(key)==self._rank)
        except AssertionError:
            sys.exit('SparseTensor.__addkey__(key): len(key)!=self._rank')
        for n in range(self._rank):
            if key[n] not in self._keys[n]:
                self._keys[n][key[n]]=[key]
            elif key[n] in self._keys[n]:
                if key not in self._keys[n][key[n]]:
                    self._keys[n][key[n]].append(key)

    #resets self._keys and fills in values consistent with the current self._tensor
    def __updatekeys__(self):
        self._keys=[dict() for n in range(self._rank)]
        for key in self._tensor.keys():
            self.__addkey__(key)

    #return a dict_view of all keys on index
    def __getkeys__(self,index):
        return self._keys[index]
    
    def __getitem__(self,ind):
        return self._tensor[ind]

    #inserts "value" into the class (not copied), and updates _Ds and _keys according to key and shape
    #shape and key are both (nested) tuples that define the shape and the quantum number of "value".
    #Note that for nested tuples, like e.g. shape=((2,3),(2,(3,4)),5), the tensor shape has to be value.shape=(6,24,5); 
    #this is currently not checked and the user has thus to provide the correct tensor. Inserting a wrong size may brake the code
    #at any point
    def __insertmergedblock__(self,key,shape,value):
        try:
            assert(len(key)==len(shape))
            assert(len(key)==self._rank)

        except AssertionError:
            sys.exit('in SparseTensor.__insertmergedblock__(self,key,shape,value): an one of the following assertions failed: (1) assert(len(key)==len(shape)); (2) assert(len(key)==self._rank)')            

        #self._tensor[key]=np.copy(value)
        self._tensor[key]=value
        self.__addkey__(key)
        self._shapes[key]=shape
        for n in range(len(key)):
            self.__insertshape__(key[n],shape[n],n)

    #updates self._Ds[index] using key and shape
    def __insertshape__(self,key,shape,index):
        #assert(len(key)==len(shape))
        if key in self._Ds[index].keys():
            del self._Ds[index][key]
            self._Ds[index][key]=shape            
        elif key not in self._Ds[index].keys():
            self._Ds[index][key]=shape
            
    #inserts a reference of "value" at "key"; the "key" is added to the self._keys list, self._Ds is updated according to value.shape
    #this function only works for tensor with a trivial mergelevel, i.e. self._mergelevel=['A']*self._rank; 
    #for merged tensors, use __insertmergedblock__ instead
    def __setitem__(self,key,value):
        assert(self._mergelevel==tuple(['A']*self._rank))
        for n in range(self._rank):
            if key[n] in self._Ds[n]:
                try :
                    assert(utils.prod(utils.flatten(self._Ds[n][key[n]]))==value.shape[n])
                except AssertionError:
                    print (n,'key=',key,'dim of k[',n,']=',utils.prod(utils.flatten(self._Ds[n][key[n]])),'value.shape: ',value.shape)
                    sys.exit('SparseTensor.__setitem__(key,value): value.shape not consistent with existing blocks')
            elif key[n] not in self._Ds[n]:
                self._Ds[n][key[n]]=value.shape[n]

        #self._tensor[key]=np.copy(value)
        self._tensor[key]=value
        self._shapes[key]=value.shape
        self.__addkey__(key)

    #remove a key from the tensor
    def __remove__(self,key):
        #first remove the tensor from the dictionary
        if key in self._tensor:
            del self._tensor[key]
        if key in self._shapes:            
            del self._shapes[key]
        #now update self._keys and self._Ds:
        for n in range(len(self._keys)):
            if key in self._keys[n][key[n]]:
                self._keys[n][key[n]].remove(key)
                #if key was the only key with a quantum number key[n] on index n, then remove (key[n],self._keys[n]) from the dictionary at index n
                #also remove the corresponding entry in (key[n],self._Ds[n][key[n]]), which gives the bond-dimension of all blocks with quantum number key[n] at index n
                #since key was the only entry with a quantum number key[n] at index n, it has to be removed from self._Ds[n]
                if len(self._keys[n][key[n]])==0:
                    del self._Ds[n][key[n]]
                    del self._keys[n][key[n]]
        
    #removes all blocks with a total norm smaller then thresh/D**rank
    def __squeeze__(self,thresh=1E-14):
        toberemoved=list()
        for k in self._tensor.keys():
            if np.linalg.norm(self[k])/utils.prod(utils.flatten(self[k].shape))<thresh:
                toberemoved.append(k)
        for k in toberemoved:
            self.__remove__(k)
        return self
    
    #for use with print command
    def __str__(self):
        for k in self._tensor.keys():
            print 
            print 
            print (k)
            print (self._tensor[k])
        return ''

    def __keys__(self):
        return self._tensor.keys()
    
    def __blockmax__(self):
        mx=dict()
        for c in self.__keys__():
            mx[c]=np.max(self[c])
        return mx
    
    def __blocknorms__(self):
        norm=dict()
        for c in self.__keys__():
            norm[c]=np.linalg.norm(self[c])
        return norm

    def __norm__(self):
        Z=list(snp.tensordot(self,snp.conj(self),(range(self._rank),range(self._rank)))._tensor.values())[0]
        return np.sqrt(Z)
    def __normalize__(self):
        Z=list(snp.tensordot(self,snp.conj(self),(range(self._rank),range(self._rank)))._tensor.values())[0]
        for k in self._tensor:
            t=self._tensor[k]
            self._tensor[k]=t/Z



    #act on all blocks with "operation"
    def __unary_operations__(self,operation,*args,**kwargs):
        res=self.__copy__()
        res._defval=operation(self._defval,*args,**kwargs)
        if res._defval!=float(0.0) and res._defval!=complex(0.0):

            warnings.warn('SparseTensor.__unary_operations__(self,operation,*args,**kwargs) resulted in non-invariant tensor',stacklevel=3)
        
        for k,t in self._tensor.items():
            res._tensor[k]=operation(t,*args,**kwargs)
        return res
    
    #act with operation on all blocks of self._tensor and obj._tensor
    #binary operatrions include multiplication with scalars, so a could be
    #a number of a tensor. This functions is taken from Markus Haurus abeliantensor.py
    #code
    def __binary_operations__(self,a,operation,*args,**kwargs):
        res=self.__copy__()
        try:
            if np.result_type(a._dtype,self._dtype)==np.float64:
                res._dtype=float
            elif np.result_type(a._dtype,self._dtype)==np.complex128:
                res._dtype=complex
            else:
                sys.exit(' __binary_operations__(self,a,operation,*args,**kwargs): unknown dtype',np.result_type(a._dtype,self._dtype))
        except AttributeError:
            if np.result_type(a,self._dtype)==np.float64:
                res._dtype=float
            elif np.result_type(a,self._dtype)==np.complex128:
                res._dtype=complex
            else:
                sys.exit(' __binary_operations__(self,a,operation,*args,**kwargs): unknown dtype',np.result_type(a._dtype,self._dtype))

        if isinstance(a,SparseTensor):
            #do some checks:
            if self._qflow!=a._qflow:
                warnings.warn('SparseTensor.__binary_operations__(self,a,operation,*args,**kwargs): tensors have different qflow; using qflow of self',stacklevel=3)
            if self._mergelevel!=a._mergelevel:
                sys.exit('SparseTensor.__binary_operations__(self,a,operation,*args,**kwargs): tensors have different mergelevel') 

            res._defval=operation(self._defval,a._defval,*args,**kwargs)
            keys = set().union(self._tensor.keys(), a._tensor.keys())
            if self._mergelevel==tuple(['A']*self._rank):
                for k in keys:
                    temp1 = self._tensor.get(k, self._defval)
                    temp2 = a._tensor.get(k, a._defval)
                    res[k]=operation(temp1, temp2, *args, **kwargs)
                
            elif self._mergelevel!=tuple(['A']*self._rank):
                for k in keys:
                    temp1 = self._tensor.get(k, self._defval)
                    temp2 = a._tensor.get(k, a._defval)
                    try:
                        shape=self._shapes[k]
                    except KeyError:
                        try:
                            shape=a._shapes[k]
                        except KeyError:
                            sys.exit('SparseTensor.__in_place_binary_operations(): KeyError')
                    res.__insertmergedblock__(k,shape,operation(temp1, temp2, *args, **kwargs))

        else:
            res._defval = operation(self._defval,a,*args, **kwargs)
            for k,v in self._tensor.items():
                res._tensor[k] = operation(v, a, *args, **kwargs)
            
        if res._defval!=float(0.0) and res._defval!=complex(0.0) and res._defval!=True:
            warnings.warn('SparseTensor.__binary_operations__(self,operation,*args,**kwargs) resulted in non-invariant tensor',stacklevel=3)

        return res


    def __in_place_binary_operations__(self,a,operation,*args,**kwargs):
        try:
            if np.result_type(a._dtype,self._dtype)==np.float64:
                self._dtype=float
            elif np.result_type(a._dtype,self._dtype)==np.complex128:
                self._dtype=complex
            else:
                sys.exit(' __in_place_binary_operations__(self,a,operation,*args,**kwargs): unknown dtype',np.result_type(a._dtype,self._dtype))
        except AttributeError:
            if np.result_type(a,self._dtype)==np.float64:
                self._dtype=float
            elif np.result_type(a,self._dtype)==np.complex128:
                self._dtype=complex
            else:
                sys.exit(' __in_place_binary_operations__(self,a,operation,*args,**kwargs): unknown dtype',np.result_type(a._dtype,self._dtype))

        if isinstance(a,SparseTensor):
            #do some checks: mergelevels of a and self have to be identical
            #print 'in binary-inplace add: ',self._qflow,a._qflow
            if self._qflow!=a._qflow:
                warnings.warn('SparseTensor.__binary_operations__(self,a,operation,*args,**kwargs): tensors have different qflow; using qflow of self',stacklevel=3)
            if self._mergelevel!=a._mergelevel:
                sys.exit('SparseTensor.__binary_operations__(self,a,operation,*args,**kwargs): tensors have different mergelevel') 

            self._defval=operation(self._defval,a._defval,*args,**kwargs)
            keys = set().union(self._tensor.keys(), a._tensor.keys())

            if self._mergelevel==tuple(['A']*self._rank):
                for k in keys:
                    temp1 = self._tensor.get(k, self._defval)
                    temp2 = a._tensor.get(k, a._defval)
                    self[k]=operation(temp1, temp2, *args, **kwargs)
                
            elif self._mergelevel!=tuple(['A']*self._rank):
                for k in keys:
                    temp1 = self._tensor.get(k, self._defval)
                    temp2 = a._tensor.get(k, a._defval)
                    try:
                        shape=self._shapes[k]
                    except KeyError:
                        try:
                            shape=a._shapes[k]
                        except KeyError:
                            sys.exit('SparseTensor.__in_place_binary_operations(): KeyError')
                    self.__insertmergedblock__(k,shape,operation(temp1, temp2, *args, **kwargs))
        else:
            self._defval = operation(self._defval,a,*args, **kwargs)
            for k,v in self._tensor.items():
                self._tensor[k] = operation(v, a, *args, **kwargs)
        if self._defval!=float(0.0) and self._defval!=complex(0.0) and self._defval!=True:
            warnings.warn('SparseTensor.__in_place_binary_operations__(self,operation,*args,**kwargs) resulted in non-invariant tensor',stacklevel=3)            

        return self
    
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
    
    """
    returns a charge-diagonal identity matrix which can be contracted with "index".
    "which" specifies which index should be contractable with self._tensor[index]
    The eye._qflow on the first index is the flipped version of self._qflow[index]
    

    """
    def __eye__(self,index, which=0):
        if which>1:
            sys.exit('SparseTensor.__eye__(self,index={0},which={1}): which has be to be 0 or 1.'.format(index,which))
        Ds=[copy.deepcopy(self._Ds[index]),copy.deepcopy(self._Ds[index])]
        if which==0:
            qflow=tuple([utils.flipsigns(self._qflow[index]),self._qflow[index]])
        elif which==1:
            qflow=tuple([self._qflow[index],utils.flipsigns(self._qflow[index])])
            
        mergelevel=tuple([self._mergelevel[index],self._mergelevel[index]])
        keytoq=tuple([self._ktoq[index],self._ktoq[index]] )       
        iden=SparseTensor(keys=[],values=[],Ds=Ds,qflow=qflow,keytoq=keytoq,mergelevel=mergelevel,dtype=self._dtype)
        for k in self._Ds[index].keys():
            key=tuple([k,k])
            temp=self._Ds[index][k]
            shape=tuple([temp,temp])
            size=utils.prod(utils.flatten(temp))
            iden.__insertmergedblock__(key,shape,np.eye(size).astype(self._dtype))
            
        return iden


    #returns a random charge-diagonal matrix that can be contracted with "index"    
    def __random__(self,index,which=0):
        if which>1:
            sys.exit('SparseTensor.__eye__(self,index={0},which={1}): which has be to be 0 or 1.'.format(index,which))
        Ds=[copy.deepcopy(self._Ds[index]),copy.deepcopy(self._Ds[index])]
        if which==0:
            qflow=tuple([utils.flipsigns(self._qflow[index]),self._qflow[index]])
        elif which==1:
            qflow=tuple([self._qflow[index],utils.flipsigns(self._qflow[index])])
        
        mergelevel=tuple([self._mergelevel[index],self._mergelevel[index]])
        keytoq=tuple([self._ktoq[index],self._ktoq[index]])               
        iden=SparseTensor(keys=[],values=[],Ds=Ds,qflow=qflow,keytoq=keytoq,mergelevel=mergelevel,dtype=self._dtype)
        for k in self._Ds[index].keys():
            key=tuple([k,k])
            temp=self._Ds[index][k]
            shape=tuple([temp,temp])
            size=utils.prod(utils.flatten(temp))
            iden.__insertmergedblock__(key,shape,np.random.rand(size,size).astype(self._dtype))
            
        return iden

    #return a deep copy of self
    def __copy__(self):
        copy=SparseTensor(keys=[],values=[],Ds=self._Ds,qflow=self._qflow,keytoq=self._ktoq,mergelevel=self._mergelevel,\
                          dtype=self._dtype)        
        for k in self.__keys__():
            copy._tensor[k]=np.copy(self[k])
        #update keys makes sure that the tensor has consistent shapes
        copy.__updatekeys__()
        copy._shapes=copy.__shapes__()
        return copy

    """
    upon conjugation, the flow-sign of the tensor is changed to minus its flowsign;
    this is neccessary when using SparseTensor for DMRG
    """
    def __conjugate__(self,*args,**kwargs):
        res=self.__unary_operations__(np.conj,*args,**kwargs)
        res._qflow=utils.flipsigns(res._qflow)        
        return res
    
    def __conj__(self,*args,**kwargs):
        res=self.__unary_operations__(np.conj,*args,**kwargs)
        res._qflow=utils.flipsigns(res._qflow)                
        return res
    #take elementwise exp
    def __exp__(self,*args,**kwargs):
        res=self.__unary_operations__(np.exp,*args,**kwargs)
        return res
        
    #take elementwise sqrt
    def __sqrt__(self,*args,**kwargs):
        res=self.__unary_operations__(np.sqrt,*args,**kwargs)
        return res

    def __real_(self,*args,**kwargs):
        res=self.__unary_operations__(np.real,*args,**kwargs)
        res._dtype=float        
        return res

    def __imag_(self,*args,**kwargs):
        res=self.__unary_operations__(np.imag,*args,**kwargs)
        res._dtype=float        
        return res

    #transpose the tensor
    def __transpose__(self,newinds):
        if newinds==list(range(self._rank)):
            return self.__copy__()
        else:
            try: 
                assert(len(newinds)==self._rank)
                Ds=[dict() for n in range(self._rank)]
                for n in range(len(newinds)):
                    Ds[n]=copy.deepcopy(self._Ds[newinds[n]])
                qflow=tuple([])
                mergelevel=tuple([])
                ktoq=tuple([])
                for i1 in newinds:
                    qflow+=tuple([self._qflow[i1]])
                    mergelevel+=tuple([self._mergelevel[i1]])
                    ktoq+=tuple([self._ktoq[i1]])
            
                result=SparseTensor(keys=[],values=[],Ds=Ds,qflow=qflow,keytoq=ktoq,mergelevel=mergelevel,dtype=self._dtype)
                for key in self.__keys__():
                    result._tensor[tuple([key[n] for n in newinds])]=np.transpose(self[key],newinds)

                result.__updatekeys__()
                result._shapes=result.__shapes__()
                return result
            except AssertionError:
                sys.exit('ERROR in SparseTensor.py: transpose(tensor,newinds): len(newinds)!=tensor._rank')

    def __trace__(self):
        try:
            assert(self._rank==2)
        except AssertionError:
            sys.exit('in SparseTensor.__trace__(self,tensor): self is not a rank 2 tensor')
        tr=self._dtype(0.0)
        for k in self.__keys__():
            if k[0]!=k[1]:
                continue
            tr+=np.trace(self[k])
        return tr
    #return the hermitian conjugate, if self is a matrix
    def __herm__(self):
        try:
            assert(self._rank==2)
        except AssertionError:
            sys.exit('in SparseTensor.__herm__(self,tensor): self is not a rank 2 tensor')
            
        newinds=[1,0]
        Ds=[dict() for n in range(self._rank)]
        for n in range(len(newinds)):
            Ds[n]=self._Ds[newinds[n]].copy()
        qflow=tuple([])
        mergelevel=tuple([])
        ktoq=tuple([])        
        for i1 in newinds:
            qflow+=tuple([self._qflow[i1]])
            mergelevel+=tuple([self._mergelevel[i1]])
            ktoq=+tuple([self._ktoq[i1]])
        result=SparseTensor(keys=[],values=[],Ds=Ds,qflow=qflow,keytoq=ktoq,mergelevel=mergelevel,dtype=self._dtype)
        for key in self.__keys__():
            result._tensor[tuple([key[n] for n in newinds])]=np.conj(np.transpose(self[key],newinds))
        result.__updatekeys__()
        return result
                
    def __tondarray__(self):
        sizes=tuple([])
        blocks=[dict() for n in range(self._rank)]
        for n in range(self._rank):
            s=0
            start=0
            for k in sorted(self._Ds[n].keys()):
                blocks[n][k]=tuple([start,start+utils.prod(utils.flatten(self._Ds[n][k]))])
                start+=utils.prod(utils.flatten(self._Ds[n][k]))
                s+=utils.prod(utils.flatten(self._Ds[n][k]))
            sizes+=tuple([s])
        
        full=np.zeros(sizes,dtype=self._dtype)
        for k in self.__keys__():
            b=tuple([])
            for n in range(self._rank):
                b+=tuple([slice(blocks[n][k[n]][0],blocks[n][k[n]][1],1)])
            full[b]=self[k]
        return full

    def __fromndarray__(self,array):
        sizes=tuple([])
        blocks=[dict() for n in range(self._rank)]
        for n in range(self._rank):
            s=0
            start=0
            for k in sorted(self._Ds[n].keys()):
                blocks[n][k]=tuple([start,start+utils.prod(utils.flatten(self._Ds[n][k]))])
                start+=utils.prod(utils.flatten(self._Ds[n][k]))
                s+=utils.prod(utils.flatten(self._Ds[n][k]))
            sizes+=tuple([s])
            
        for k in self.__keys__():
            b=tuple([])
            for n in range(self._rank):
                b+=tuple([slice(blocks[n][k[n]][0],blocks[n][k[n]][1],1)])
            self._tensor[k]=full[b]
        

