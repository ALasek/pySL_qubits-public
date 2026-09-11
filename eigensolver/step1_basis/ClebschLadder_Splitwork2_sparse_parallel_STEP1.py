"""STEP1 of the Heisenberg paper workflow.

This script builds basis states in the coupled-spin (`S`, `M`) basis using a
Clebsch-Gordan ladder/tree construction. The resulting basis-state pickles in
`../statesSMparts/` and `../statesSM/` are consumed by
`HeisenbergNNN_BlockDiag_optim_splitM_savebyM_STEP2.py`.
"""

import numpy as np
from numpy import linalg as LA
import pickle
from qutip.utilities import clebsch
import copy
from scipy.io import savemat
from scipy.io import loadmat
from scipy import sparse as sp

import multiprocessing
import concurrent.futures
import time
import matplotlib.pyplot as plt
import psutil, os
import glob
import ctypes

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
 
        

def get_total_memory_usage():
    # Get the memory usage for the current process
    process = psutil.Process()
    memory_usage = process.memory_info().rss

    # Get the memory usage for all child processes
    for child in process.children(recursive=True):
        memory_usage += child.memory_info().rss

    return memory_usage  
    
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
                
            
def buildPsiTree(tree,psis, bstring, c,depth,routejs,maxqubits):
    
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
        
        for child in tree.children:
            
            routejsX=copy.deepcopy(routejs)
            routejsX.append(child.data[0])

            buildPsiTree((child),psis, (bstring+a), ci,(depth+1),(routejsX),maxqubits)
                # else:

                    

def main_process(target,maxqubits):
    
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
    
    
    # bstring=''
    
    # nodeCount=np.ones(1,dtype=np.int32)
    
    # tree=Node(1,None,1)
    # buildAmpsTree(out,targetS,targetM,maxqubits,tree,nodeCount)
    # buildPsiTree((tree),psis, bstring, 1,1,[targetS],maxqubits)
    
    

    psisout=[]
    
    # for psi in psis:
    #     psisout.append(psi.data)

        
    
    
    # for i in range(0,len(psis)):
    #     psisout.append(psis[i].data.toarray()[0])
        
    # processMem = psutil.Process(os.getpid())
    # print("MemUsage of:")
    # print(target)
    # print(processMem.memory_info().rss/(1024**2))
    # print(str(processMem.memory_percent()) + ' %')
    
    # with open('..//statesSM//States_Q'+str(maxqubits)+'_S='+str(targetS)+'_M='+str(targetM)+'.pickle', 'wb') as f:
    #     pickle.dump(psisout,f)
        
def writeparts(countdict,partsdict,psis,S,M,maxqubits):
    pd=partsdict[(S,M)]
    with open('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(S)+'_M='+str(M)+'TEMPpart'+str(pd)+'.pickle', 'wb') as f:
        pickle.dump(psis,f)
    os.rename('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(S)+'_M='+str(M)+'TEMPpart'+str(pd)+'.pickle','..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(S)+'_M='+str(M)+'part'+str(pd)+'.pickle')
    partsdict[(S,M)]+=1
    countdict[(S,M)]=0
       

def collectparts(partsdict,maxqubits,targets):
    for target in targets:
        psisout=[]
        psipartial={}
        partsnr=partsdict[(target[0],target[1])]
        for part in range(0,partsnr+1):
            with open('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0part'+str(part)+'.pickle', 'rb') as f:
                psis= pickle.load(f)
                for item in list(psis.items()):
                    if item[0] in psipartial:
                        psipartial[item[0]]+=item[1]
                    else:
                        psipartial[item[0]]=item[1]
        for value in list(psipartial.values()):
            psisout.append(value)
        with open('..//statesSM//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'_M='+str(target[1])+'.pickle', 'wb') as f:
            pickle.dump(psisout,f)

def collectparts_final(maxqubits,targets):
    for target in targets:
        if target[1]>=0:
            psisout=[]
            
            with open('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0collector.pickle', 'rb') as f:
                psis= pickle.load(f)
                for item in list(psis.items()):
                    psisout.append(item[1])
    
            with open('..//statesSM//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'_M='+str(target[1])+'.pickle', 'wb') as f:
                pickle.dump(psisout,f)
                
            os.remove('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0collector.pickle')
        

    
            
def collectparts_thrV2(maxqubits,targets,shared_collidle,shared_targetlock,partmin,partlimit,timeout):  

    log=True
    while True:
        time.sleep(timeout)
        shared_collidle.value=shared_collidle.value+1
        if log:
            with open('..//statesSMparts//collectthrlog.txt', 'a') as file:
                file.write("shared collide = "+str(shared_collidle.value)+" \n")
        
        for target in targets:
            
            files = glob.glob('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0part*.pickle')
            
            if len(files)>0:
                if len(files)>partmin:
                    if log:
                        with open('..//statesSMparts//collectthrlog.txt', 'a') as file:
                            file.write("collecting target="+str(target)+' with ' +str(len(files))+" files \n")
                           
                    shared_collidle.value=0
                    
                    filecount=0
                    psislist=[]
                    for file in files:
                        if filecount>=partlimit:
                            break
                        while True:
                            if   os.path.getsize(file) > 0  : #shared_targetlock.value != (str((target[0],target[1]))).encode('utf-8') and
                                with open(file, 'rb') as f:
                                    #print(file)
                                    psislist.append(pickle.load(f))
                                    filecount+=1
                                break
                            else:
                                time.sleep(0.5)
                        os.remove(file)
                
                    if os.path.isfile('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0collector.pickle'):
                        with open('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0collector.pickle', 'rb') as f:
                            psipartial= pickle.load(f)
                    else:
                        psipartial={}
                        
                    for psis in psislist:
                        for item in list(psis.items()):
                            if item[0] in psipartial:
                                psipartial[item[0]]+=item[1]
                            else:
                                psipartial[item[0]]=item[1]
                                    
     
                    with open('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0collector.pickle', 'wb') as f:
                        pickle.dump(psipartial,f)
                        
def collectparts_thrV2Final(maxqubits,targets,shared_collidle):  

    log=True
   
    time.sleep(1)
    shared_collidle.value=shared_collidle.value+1
    if log:
        with open('..//statesSMparts//collectthrlog.txt', 'a') as file:
            file.write("shared collide = "+str(shared_collidle.value)+" \n")
    
    for target in targets:
        
        files = glob.glob('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0part*.pickle')
        
        if len(files)>0:
            if True:
                if log:
                    with open('..//statesSMparts//collectthrlog.txt', 'a') as file:
                        file.write("collecting target="+str(target)+' with ' +str(len(files))+" files \n")
                       
                shared_collidle.value=0
                
                filecount=0
                psislist=[]
                for file in files:
                    while True:
                        if   os.path.getsize(file) > 0  : #shared_targetlock.value != (str((target[0],target[1]))).encode('utf-8') and
                            with open(file, 'rb') as f:
                                #print(file)
                                psislist.append(pickle.load(f))
                                filecount+=1
                            break
                        else:
                            time.sleep(0.5)
                    os.remove(file)
            
                if os.path.isfile('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0collector.pickle'):
                    with open('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0collector.pickle', 'rb') as f:
                        psipartial= pickle.load(f)
                else:
                    psipartial={}
                    
                for psis in psislist:
                    for item in list(psis.items()):
                        if item[0] in psipartial:
                            psipartial[item[0]]+=item[1]
                        else:
                            psipartial[item[0]]=item[1]
                                
 
                with open('..//statesSMparts//StatesSplitwork_Q'+str(maxqubits)+'_S='+str(target[0])+'.0_M='+str(target[1])+'.0collector.pickle', 'wb') as f:
                    pickle.dump(psipartial,f)
    
if __name__ == "__main__":

    TSTART=time.time()
    
    Check=False
    maxqubits=14
    timeout=10
    filelimit=6
    partmin=3
    
    timer=0
    maxS=maxqubits//2
    targets=[]
    for i in range(0,maxS+1):
        for j in range(0, 2*i+1):
            targets.append([i,-i+j])
    
    # targets=[[4,-4],[4,-3],[4,-2],[4,-1],[4,0],[4,1],[4,2],[4,3],[4,4]]
    #targets=[[5,-5],[5,-4],[5,-3],[5,-2],[5,-1],[5,0],[5,1],[5,2],[5,3],[5,4],[5,5]]
    #targets=[[6,-3],[6,-2],[6,-1],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]
    
    
    
    
    
    
    # targetstart=6
    

    
    
    print(len(targets))    
    

    # print(target)
    
    # j1=0.5
    
    # j2=0.5
    
    # m1R=np.arange(-j1,j1+1,1)
    # m2R=np.arange(-j2,j2+1,1)
    
    # j3R=np.arange((j1+j2)%1,j1+j2+1,1)
    
    
    
    psis={}
    
    psis[1]={}
    
    #psis[1][(0.5,0.5)]=[1,0]
    psis[1][(0.5,0.5,1)]={(0.5,0):sp.csc_matrix([1,0],dtype=np.float32)}
    #psis[1][(0.5,-0.5)]=[0,1]
    psis[1][(0.5,-0.5)]={(0.5,0):sp.csc_matrix([0,1],dtype=np.float32)}
    
   
    writelimit=2000
    countdict={}
    partsdict={}
    for target in targets:
        countdict[(target[0],target[1])]=0
        partsdict[(target[0],target[1])]=0
    
  
    
    ns=[2]
    
    while True:
        newns=2*ns[-1]
        if newns<= maxqubits:
            ns.append(newns)
        else:
            break
    
    
    for n in ns:
    
        print('Doing n='+str(n))
        nhalf=n//2
        psis[n]={}
        
        getkeys=list(psis[nhalf].keys())
        
        for key1 in getkeys:
            for key2 in getkeys:
                j1=key1[0]
                j2=key2[0]
                
                m1=key1[1]
                m2=key2[1]
                
                #goodjs=np.arange((j1+j2)%1,j1+j2+1,1)
                goodjs=np.arange(abs(j1-j2),j1+j2+1,1)
        
                for j3 in goodjs:
                    if abs(m1+m2)<=j3:
                        cb = clebsch(j1, j2, j3, m1, m2, m1+m2)
                        if abs(cb)>0:
                            if (j3,m1+m2) not in psis[n]:
                                psis[n][(j3,m1+m2)]={}
     
                            
                            for psis1 in list(psis[nhalf][key1].items()):
                                for psis2 in list(psis[nhalf][key2].items()):
                                   seqkey=((j1,j2))+(psis1[0],psis2[0]) 
                                   if seqkey in psis[n][(j3,m1+m2)]:
                                       psis[n][(j3,m1+m2)][seqkey ]+= cb*sp.kron(psis1[1],psis2[1])
                                   else:
                                       #countdict[(j3,m1+m2)]+=1
                                       psis[n][(j3,m1+m2)][seqkey ]= cb*sp.kron(psis1[1],psis2[1])
                                   
            
            
            
        print('finished n='+str(n))
        
        
# 8+4=12 +2=14
        
    # n=16
    # n1=8
    # n2=8
    # psis[n]={}
    
    # getkeys16=list(psis[n1].keys())
    # getkeys2=list(psis[n2].keys())
    
    
    # for key1 in getkeys16:
    #     for key2 in getkeys2:
    #         j1=key1[0]
    #         j2=key2[0]
            
    #         m1=key1[1]
    #         m2=key2[1]
            
    #         #goodjs=np.arange((j1+j2)%1,j1+j2+1,1)
    #         goodjs=np.arange(abs(j1-j2),j1+j2+1,1)
    
    #         for j3 in goodjs:
    #             if abs(m1+m2)<=j3:
    #                 cb = clebsch(j1, j2, j3, m1, m2, m1+m2)
    #                 if abs(cb)>0:
    #                     if (j3,m1+m2) not in psis[n]:
    #                         psis[n][(j3,m1+m2)]={}
 
                        
    #                     for psis1 in list(psis[n1][key1].items()):
    #                         for psis2 in list(psis[n2][key2].items()):
    #                             seqkey=((j1,j2))+(psis1[0],psis2[0]) 
    #                             if seqkey in psis[n][(j3,m1+m2)]:
    #                                 psis[n][(j3,m1+m2)][seqkey ]+= cb*sp.kron(psis1[1],psis2[1])
    #                             else:
    #                                 #countdict[(j3,m1+m2)]+=1
    #                                 psis[n][(j3,m1+m2)][seqkey ]= cb*sp.kron(psis1[1],psis2[1])
    #                                 # if countdict[(j3,m1+m2)]>writelimit:
    #                                 #     writeparts(countdict,partsdict, psis[n][(j3,m1+m2)], j3, m1+m2,maxqubits)
    #                                 #     psis[n][(j3,m1+m2)]={}
                                        
           
            
           
    if maxqubits==14:
        
        ns.append(12)
        
        n=12
        
        print('Doing n='+str(n))
        n1=4
        n2=8
        psis[n]={}
        
        getkeys1=list(psis[n1].keys())
        getkeys2=list(psis[n1].keys())
        
        for key1 in getkeys1:
            for key2 in getkeys2:
                j1=key1[0]
                j2=key2[0]
                
                m1=key1[1]
                m2=key2[1]
                
                #goodjs=np.arange((j1+j2)%1,j1+j2+1,1)
                goodjs=np.arange(abs(j1-j2),j1+j2+1,1)
        
                for j3 in goodjs:
                    if abs(m1+m2)<=j3:
                        cb = clebsch(j1, j2, j3, m1, m2, m1+m2)
                        if abs(cb)>0:
                            if (j3,m1+m2) not in psis[n]:
                                psis[n][(j3,m1+m2)]={}
     
                            
                            for psis1 in list(psis[n1][key1].items()):
                                for psis2 in list(psis[n2][key2].items()):
                                   seqkey=((j1,j2))+(psis1[0],psis2[0]) 
                                   if seqkey in psis[n][(j3,m1+m2)]:
                                       psis[n][(j3,m1+m2)][seqkey ]+= cb*sp.kron(psis1[1],psis2[1])
                                   else:
                                       #countdict[(j3,m1+m2)]+=1
                                       psis[n][(j3,m1+m2)][seqkey ]= cb*sp.kron(psis1[1],psis2[1])
           
            
    shared_collidle = multiprocessing.Value('i', 0)
    shared_targetlock = multiprocessing.Array(ctypes.c_char, 100)

    collectprocess = multiprocessing.Process(target=collectparts_thrV2, args=(maxqubits,targets,shared_collidle,shared_targetlock,partmin,filelimit,timeout)) 
    collectprocess.daemon=True
    collectprocess.start()
            
   
    n=maxqubits
    # n1=16
    # n2=4
    n1=ns[-1]
    for ntry in ns:
        if ntry+n1==n:
            n2=ntry
            break
        
    psis[n]={}
    
    getkeys16=list(psis[n1].keys())
    getkeys2=list(psis[n2].keys())
    
    
    for key1 in getkeys16:
        for key2 in getkeys2:
            j1=key1[0]
            j2=key2[0]
            
            m1=key1[1]
            m2=key2[1]
            
            #goodjs=np.arange((j1+j2)%1,j1+j2+1,1)
            goodjs=np.arange(abs(j1-j2),j1+j2+1,1)
    
            for j3 in goodjs:
                # print('Doing j3='+str(j3))
                if abs(m1+m2)<=j3 and (m1+m2)>=0:
                    cb = clebsch(j1, j2, j3, m1, m2, m1+m2)
                    if abs(cb)>0:
                        if (j3,m1+m2) not in psis[n]:
                            psis[n][(j3,m1+m2)]={}
 
                        
                        for psis1 in list(psis[n1][key1].items()):
                            for psis2 in list(psis[n2][key2].items()):
                                seqkey=((j1,j2))+(psis1[0],psis2[0]) 
                                if seqkey in psis[n][(j3,m1+m2)]:
                                    psis[n][(j3,m1+m2)][seqkey ]+= cb*sp.kron(psis1[1],psis2[1])
                                else:
                                    countdict[(j3,m1+m2)]+=1
                                    psis[n][(j3,m1+m2)][seqkey ]= cb*sp.kron(psis1[1],psis2[1])
                                    if countdict[(j3,m1+m2)]>writelimit:
                                        shared_targetlock.value=str((j3,m1+m2)).encode('utf-8')
                                        writeparts(countdict,partsdict, psis[n][(j3,m1+m2)], j3, m1+m2,maxqubits)
                                        shared_targetlock.value=str(0).encode('utf-8')
                                        psis[n][(j3,m1+m2)]={}
                                        
                         
                                
    print('start writeparts')
    for key in list(psis[n].keys()):
        shared_targetlock.value=str((key[0],key[1])).encode('utf-8')
        writeparts(countdict,partsdict, psis[n][key], key[0], key[1],maxqubits)
        shared_targetlock.value=str(0).encode('utf-8')
        
    
    #collectparts(partsdict,maxqubits,targets)
    print('start collect')
    while True:
        time.sleep(15)
        if shared_collidle.value>=2:
            collectprocess.terminate()
            break
        
    print('start collect prefinal')
    collectparts_thrV2Final(maxqubits,targets,shared_collidle)
    print('start collect final')
    collectparts_final(maxqubits,targets)
    


    TEND=time.time()
    print("Time elapsed:")
    print(str(TEND-TSTART))
    
