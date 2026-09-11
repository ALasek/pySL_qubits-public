# -*- coding: utf-8 -*-
"""
Created on Thu May 19 02:02:05 2022

@author: OlekAdmin
"""

import numpy as np
from numpy import linalg as LA
import pickle
from qutip.utilities import clebsch
import copy
from scipy.io import savemat
from scipy.io import loadmat
from scipy import sparse as sp
import time, sys

import matplotlib.pyplot as plt

class Psi:
    def __init__(self, data,j):
        self.data=data
        self.j=j


class Node:
    def __init__(self, data, parent, counter):
        
        if parent==None:
            self.root=True
        else:
            self.root=False
            
        self.id=counter
        self.children = []
        self.parent = parent
        self.data = data

# Insert Node
    def insert(self, data,counter):
       newchild=Node(data,self,counter)
       self.children.append(newchild)
 
        
 
def pauli_sum(sigma,n):
    #works only for 5 or more qubits
    outvar=np.zeros((2**n,2**n))+(1j)*np.zeros((2**n,2**n), dtype=np.complex64)
    for ll in range(n):
        arr=[np.array([[1,0],[0,1]], dtype=np.complex64)]*n
        arr[ll]=sigma
        temp=arr[0]
        for kk in range(1,n):
            temp=np.kron(temp,arr[kk])
        outvar=outvar+temp
    return outvar

      
def buildAmpsTree(out,targetS,targetM,depth,tree,nc):
    #lastindex=np.zeros([1], dtype=np.int64)
    norm=0
        
    for row in out[depth-2]:
        if row[2]==targetS and row[5]==targetM:
            #out2[-1].append([row[0],row[3],row[4],row[6]])
            nc[0]=nc[0]+1
            tree.insert(row,nc[0])
            norm=norm+abs(row[6])**2
    if (depth-1)>1:
        for child in tree.children:
            targetS=child.data[0]
            targetM=child.data[3]
            buildAmpsTree(out,targetS,targetM,depth-1,child,nc)
                
            
def buildPsiTree(tree,psis, bstring, c,depth,routejs,splits):
    
    if len(tree.children)==0:

        if tree.data[4]==0.5:
            a='1'
        elif tree.data[4]==-0.5:
            a='0'
        else:
            print('WRONG')
            
        bstringi=bstring+a
        
        if tree.data[3]==0.5:
            a='1'
        elif tree.data[3]==-0.5:
            a='0'
        else:
            print('WRONG')
        
        found=False
        for psi in psis:
            if psi.j==routejs:
                #print(psi.j)
                data=psi.data.toarray()[0]
                data[2**maxqubits - 1 - int((bstringi+a)[::-1],2)]=data[2**maxqubits - 1 - int((bstringi+a)[::-1],2)]+c*tree.data[6] 
                psi.data=sp.csc_matrix(data)
                #psi.data[2**maxqubits - 1 - int((bstringi+a)[::-1],2)]=psi.data[2**maxqubits - 1 - int((bstringi+a)[::-1],2)]+c*tree.data[6] 
                found=True
                break
        if not found:
            psis.append(Psi(sp.csc_matrix(np.zeros(2**maxqubits,dtype=np.float64)),routejs))
            
            psi=psis[-1]

            data=psi.data.toarray()[0]
            data[2**maxqubits - 1 - int((bstringi+a)[::-1],2)]=data[2**maxqubits - 1 - int((bstringi+a)[::-1],2)]+c*tree.data[6] 
            psi.data=sp.csc_matrix(data)
            #print(psi.j)
            #psi.data[2**maxqubits - 1 - int((bstringi+a)[::-1],2)]=psi.data[2**maxqubits - 1 - int((bstringi+a)[::-1],2)]+c*tree.data[6] 

    else:

        if tree.root:
            ci=c
            a=''
        else:
            ci=c*tree.data[6]
            if tree.data[4]==0.5:
                a='1'
            elif tree.data[4]==-0.5:
                a='0'
            else:
                print('WRONG')
        
        splits[0]=splits[0]+1
        for child in tree.children:
            
            routejsX=copy.deepcopy(routejs)
            routejsX.append(child.data[0])

            buildPsiTree((child),psis, (bstring+a), ci,(depth+1),(routejsX),splits)
                # else:


Check=False
maxqubits=6


maxS=maxqubits//2
targets=[]
for i in range(0,maxS+1):
    for j in range(0, 2*i+1):
        targets.append([i,-i+j])

# targets=[[4,-4],[4,-3],[4,-2],[4,-1],[4,0],[4,1],[4,2],[4,3],[4,4]]
#targets=[[5,-5],[5,-4],[5,-3],[5,-2],[5,-1],[5,0],[5,1],[5,2],[5,3],[5,4],[5,5]]
#targets=[[6,-3],[6,-2],[6,-1],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]



# targets=[[3,2]]


targetstart=0

for iii in range(targetstart,len(targets)):
    
    # for iii in range(1):
    
        
    target=targets[iii]
    
    # target = [4,1]
        
    
    print(target)
    
    
    
    targetS=target[0]
    targetM=target[1]
    
    
    
    out=[]
    
    n=0
    
    j1=0.5
    
    j2=0.5
    
    out.append([])
    
    js=[]
    
    m1R=np.arange(-j1,j1+1,1)
    m2R=np.arange(-j2,j2+1,1)
    
    j3R=np.arange((j1+j2)%1,j1+j2+1,1)
    
    
    
    for j3 in j3R:
        for m1 in m1R:
            for m2 in m2R:
                if abs(m1+m2)<=j3:
                    c=clebsch(j1, j2, j3, m1, m2, m1+m2)
                    if abs(c)>0:
                        out[n].append([j1,j2,j3,m1,m2,m1+m2,c])
                        js.append(j3)
    
    n=1
    
    js=set(js)
    
    for i in range(1,maxqubits-1):
    
        out.append([])
        
        jsOld=js
        js=[]
        for j1 in jsOld:
            
            m1R=np.arange(-j1,j1+1,1)
            m2R=np.arange(-j2,j2+1,1)
            
            j3R=np.arange(abs(j1-j2),j1+j2+1,1)
        
        
        
            for j3 in j3R:
                for m1 in m1R:
                    for m2 in m2R:
                        if abs(m1+m2)<=j3:
                            c=clebsch(j1, j2, j3, m1, m2, m1+m2)
                            if abs(c)>0:
                                out[n].append([j1,j2,j3,m1,m2,m1+m2,c])
                                js.append(j3)
        n=n+1
        js=set(js)
        
        
     
    
    psis=[]
    
    
    bstring=''
    
    nodeCount=np.ones(1,dtype=np.int32)
    
    tree=Node(1,None,1)
    
    TSTARTamps=time.time()
    
    buildAmpsTree(out,targetS,targetM,maxqubits,tree,nodeCount)
    
    TSTARTtree=time.time()
    
    splits=np.zeros(1,dtype=np.int32)
    
    buildPsiTree((tree),psis, bstring, 1,1,[targetS],splits)
    
    TEND=time.time()
    
    print("time amps")
    print(str(TSTARTtree-TSTARTamps))
    
    print("time tree")
    print(str(TEND-TSTARTtree))
    
    if Check:
    
        eye=np.array([[1,0],[0,1]], dtype=np.complex64)
        z = np.array([[1,0],[0,-1]], dtype=np.complex64)
        x = np.array([[0,1],[1,0]], dtype=np.complex64)
        y = np.array([[0,-1j],[1j,0]], dtype=np.complex64)
        
        x_tot=(pauli_sum(x,maxqubits))*0.5
        y_tot=(pauli_sum(y,maxqubits))*0.5
        z_tot=(pauli_sum(z,maxqubits))*0.5
            
        S2=x_tot@x_tot +y_tot@y_tot + z_tot@z_tot
                
    # S2_val, S2_vec= scipy.linalg.eigh(S2)
    
    # Sz_val, Sz_vec= LA.eig(z_tot)
    psisout=[]
    
    for psi in psis:
        psisout.append(psi.data)
        if Check:
            print('S:')
            S2exp=(np.real(np.transpose(np.conjugate(psi.data)))@S2@psi.data)
            Sexp=-0.5 + 0.5*np.sqrt(1+4*S2exp)
            print(Sexp)
            print('m:')
            mexp=np.real(np.transpose(np.conjugate(psi.data)))@z_tot@psi.data
            print(mexp)
            print('S2 var')
            print(((np.real(np.transpose(np.conjugate(psi.data))@S2@S2@psi.data)))-abs(S2exp)**2)
            print('m var')
            print(np.real(np.transpose(np.conjugate(psi.data)))@z_tot@z_tot@psi.data-abs(mexp)**2)
            print('   ')
        
        
    
    
    # for i in range(0,len(psis)):
    #     psisout.append(psis[i].data.toarray()[0])
        
    
    with open('..//statesSM//States_Q'+str(maxqubits)+'_S='+str(targetS)+'_M='+str(targetM)+'.pickle', 'wb') as f:
        pickle.dump(psisout,f)
        
# len(psisout)

# mdic = {"psisout": psisout}
# savemat('Q='+str(maxqubits)+'_s='+str(targetS)+'_Eigenstates.mat',mdic)
