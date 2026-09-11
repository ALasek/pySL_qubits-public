
"""STEP2 of the Heisenberg paper workflow.

This script reads the coupled-spin basis states generated in STEP1, constructs
Hamiltonian blocks in the `S, M` basis, diagonalizes them, and writes solved
spectrum metadata plus blockwise eigenvector data into `./pickles/`.
"""

import numpy as np
from scipy import sparse as sp
from scipy.linalg import expm, logm, sqrtm
from scipy import special
from numpy import linalg as LA
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from matplotlib.font_manager import FontProperties
import math
from scipy.io import loadmat
import qutip
import csv
import random as rnd
import time
# from sklearn.metrics import r2_score
import warnings
from qutip.utilities import clebsch
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import psutil
import time

def part_tr_sp(rho,n,m):
    #trace-out all but 1st m qubits
    #up=np.array([1,0])
    #dwn=np.array([0,1])
    #t1=...np.kron(np.identity(2),np.kron(np.identity(2),up))
    #t2=...np.kron(np.identity(2),np.kron(np.identity(2),dwn))
    outvar=rho
    for ll in range(n,m,-1):
        t1=sp.lil_matrix((2**(ll-1),2**ll))
        t2=sp.lil_matrix((2**(ll-1),2**ll))
        for kk in range(2**(ll-1)):
            t1[kk,2*kk]=1
            t2[kk,2*kk+1]=1
        #endfor
        outvar=t1@outvar@t1.transpose() + t2@outvar@t2.transpose()
    #endfor
    return outvar



def measureSsingle(direction,qubitnr,nqubits):
    eye=np.array([[1,0],[0,1]])
    z = np.array([[1,0],[0,-1]])
    x = np.array([[0,1],[1,0]])
    y = np.array([[0,-1j],[1j,0]])
    out=[1]
    if direction=='x':
        m=x
    elif direction=='y':
        m=y
    elif direction=='z':
        m=z
        
    
    for i in range(0,nqubits):
        if (i)==qubitnr:
            out=np.kron(m,out)
        else:
            out=np.kron(eye,out)
    return out
   

def rotateSsingle(angle,direction,qubitnr,nqubits):
    eye=np.array([[1,0],[0,1]])
    Ry = np.array([[np.cos(angle/2),-1*np.sin(angle/2)],[np.sin(angle/2),np.cos(angle/2)]])
    Rx = np.array([[np.cos(angle/2),-1j*np.sin(angle/2)],[-1j*np.sin(angle/2),np.cos(angle/2)]])
    Rz = np.array([[np.exp(-1j * angle/2),0],[0,np.exp(1j * angle/2)]])
    out=[1]
    if direction=='x':
        Rm=Rx
    elif direction=='y':
        Rm=Ry
    elif direction=='z':
        Rm=Rz
        
    
    for i in range(0,nqubits):
        if (i)==qubitnr:
            out=np.kron(Rm,out)
        else:
            out=np.kron(eye,out)
    return out
        
def rotateSall(angle,direction,nqubits):
    eye=np.array([[1,0],[0,1]], dtype=np.complex64)
    Ry = np.array([[np.cos(angle/2),-1*np.sin(angle/2)],[np.sin(angle/2),np.cos(angle/2)]], dtype=np.complex64)
    Rx = np.array([[np.cos(angle/2),-1j*np.sin(angle/2)],[-1j*np.sin(angle/2),np.cos(angle/2)]], dtype=np.complex64)
    Rz = np.array([[np.exp(-1j * angle/2),0],[0,np.exp(1j * angle/2)]], dtype=np.complex64)
    out=[1]
    if direction=='x':
        Rm=Rx
    elif direction=='y':
        Rm=Ry
    elif direction=='z':
        Rm=Rz
        
    
    for i in range(0,nqubits):
        out=np.kron(Rm,out)

    return out
     


def pauli_sum(sigma,n):
    #works only for 5 or more qubits
    outvar=np.zeros((2**n,2**n),dtype=np.complex64)+(1j)*np.zeros((2**n,2**n), dtype=np.complex64)
    for ll in range(n):
        arr=[np.array([[1,0],[0,1]], dtype=np.complex64)]*n
        arr[ll]=sigma
        temp=arr[0]
        for kk in range(1,n):
            temp=np.kron(temp,arr[kk])
        outvar=outvar+temp
    return outvar

def pauli_sum_sp(sigma,n):
    #works only for 5 or more qubits
    outvar=sp.csr_matrix((2**n,2**n),dtype=np.complex64)
    #arr=[np.array([[1,0],[0,1]], dtype=np.complex64)]*n
    for ll in range(n):
        #arr[ll]=sigma
        #temp=sp.csr_matrix(arr[0])
        if ll==0:
            temp=sp.csr_matrix(sigma)
        else:
            temp=sp.csr_matrix(np.eye(2))
            
        for kk in range(1,n):
            if kk==ll:
                temp=sp.kron(temp,sp.csr_matrix(sigma))
            else:
                temp=sp.kron(temp,sp.csr_matrix(np.eye(2)))
        outvar=outvar+temp
    return outvar


def pauli_single(sigma,ind,n):
    #works only for 5 or more qubits
    
    arr=[np.array([[1,0],[0,1]], dtype=np.complex64)]*n
    arr[ind]=sigma
    temp=arr[0]
    for kk in range(1,n):
        temp=np.kron(temp,arr[kk])
    
    return temp


def pauli_kron(sigma,n,pair):
    arr=[np.array([[1,0],[0,1]],dtype=np.complex64)]*n
    arr[pair[0]]=sigma
    arr[pair[1]]=sigma
    outvar=arr[0]
    for ll in range(1,n):
            outvar=np.kron(outvar,arr[ll])
    return outvar


def pauli_kron_sp(sigma,n,pair):
    sigma=sp.bsr_matrix(sigma)
    i1=pair[0]
    i2=pair[1]
    
    if abs(i2-i1)>1:
        eye1=sp.eye(2**(n-i2-1),dtype=np.int8)
        eye2=sp.eye(2**(i2-i1-1),dtype=np.int8)
        eye3=sp.eye(2**(i1),dtype=np.int8)
        return sp.kron(eye3,sp.kron(sigma,sp.kron(eye2,sp.kron(sigma,eye1))))
    else:
        eye1=sp.eye(2**(n-i2-1),dtype=np.int8)
        eye3=sp.eye(2**(i2-1),dtype=np.int8)
        return sp.kron(eye3,sp.kron(sigma,sp.kron(sigma,eye1)))


def pauli_kron_two(sigma1,sigma2,n,pair):
    arr=[np.array([[1,0],[0,1]],dtype=np.complex64)]*n
    arr[pair[0]]=sigma1
    arr[pair[1]]=sigma2
    outvar=arr[0]
    for ll in range(1,n):
            outvar=np.kron(outvar,arr[ll])
    return outvar


def cfitlin(x,a,b):
    return a*np.float64(x)+b

def cfit(x,a,b):
    return a*x**b

def cfit2(x,a,b,c):
    return (a*x**b)+c

def cfitNorm(x,a,mu,var):
    return a*np.exp(-((x-mu)**2)/var)

def GOE_beta1(x,a,b):
    return a*x*np.exp(-b*x**2)

def GOE_beta2(x,a,b):
    return a*(x**2)*np.exp(-b*x**2)

def Poisson(x,a,b):
    return a*np.exp(-b*x)

def fidelity(rho,sigma):
    return np.real(np.trace(sqrtm(sqrtm(rho)@sigma@sqrtm(rho))))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return [array[idx],idx]


def norm_comm(A,B):
    return np.count_nonzero(A@B-B@A)

def CoarsegrainAvSqAlt(Xaxis,Yaxis,inputA,outputA,windowX,windowY):
    sizeX=inputA.shape[0]
    sizeY=inputA.shape[1]
    
    i=0
    j=0
    
    avsizebotX=find_nearest(Xaxis,Xaxis[sizeX//2]-windowX/2)[1]
    avsizetopX=find_nearest(Xaxis,Xaxis[sizeX//2]+windowX/2)[1]
    avsizebotY=find_nearest(Yaxis,Yaxis[sizeY//2]-windowY/2)[1]
    avsizetopY=find_nearest(Yaxis,Yaxis[sizeY//2]+windowY/2)[1]

    # avsizeX=(avsizetopX-avsizebotX)//2
    # avsizeY=(avsizetopY-avsizebotY)//2
        
    #np.mean
    prevAv=np.sum((inputA[0:avsizetopX,0:avsizetopY].ravel())**2)
    prevAv=prevAv/np.count_nonzero(inputA[0:avsizetopX,0:avsizetopY].ravel())
    outputA[0,0]=prevAv

    step=1


    # avsizebotYprev=avsizebotY
    # avsizetopYprev=avsizetopY
    # avsizebotXprev=avsizebotX
    # avsizetopXprev=avsizetopX
    for i in range(0,sizeX):
        for j in range(0,sizeY):
            #np.mean
            outputA[i,j]=np.sum(abs(inputA[max(i-avsizebotX,0):min(i+avsizebotX,sizeX-1),max(j-avsizebotY,0):min(j+avsizebotY,sizeY-1)].ravel())**2)
            outputA[i,j]= outputA[i,j]/np.count_nonzero(abs(inputA[max(i-avsizebotX,0):min(i+avsizebotX,sizeX-1),max(j-avsizebotY,0):min(j+avsizebotY,sizeY-1)].ravel()))

def RemoveFraction(frac, elements, X, Y):
    out=[]
    Xout=[]
    Yout=[]
    for e in range(0,len(elements)):
        if rnd.random()<=frac:
            out.append(elements[e])
            Xout.append(X[e])
            Yout.append(Y[e])
    return [out,Xout,Yout]
    

def mem_monitor_thr(timeout,n):
    while True:
        with open('.//logs//MEMLOG'+str(n)+'.txt', 'a') as f:
            f.write('\n')
            f.write('mem monitor, mem= '+str(psutil.virtual_memory().used/(1024**3))+' GB' )
        # print('mem monitor, mem= '+str(psutil.virtual_memory().used/(1024**3)))
        time.sleep(timeout)

def clearLog(n):
    with open('.//logs//LOG'+str(n)+'.txt', 'w') as f:
        f.write('start')
    
def writeLog(string,n):
    with open('.//logs//LOG'+str(n)+'.txt', 'a') as f:
        f.write('\n')
        f.write(str(string))


if __name__ == "__main__":
# time.sleep(60*60*3)

    
    TSTART=time.time()
    
    J=1
    #h=0.5
    #g=1.05
    
    
    
    nqubitsA=[8]
    
    memtimeout=15
    
    #nqubitsA=[6,7,8,9,10,11,12]
    
    Delta=1
    
    random_frac=0.3
    
    hist_size=35
    
    hist_size_gaps=50
    
    Bfield_z=0.00
    
    Bfield_y=0.00
    
    Bfield_x=0.00
    
    
    
    
    for kkk in range(0,len(nqubitsA)):
        
        rnd.seed(0)
        
        nqubits=nqubitsA[kkk]
            #qubits=10
        dim=2**nqubits
       
    
        
        #(y2z)u(x2y)u(z2x)u
        #"""
        #############################
        # one qubit paulis
        eye=np.array([[1,0],[0,1]], dtype=np.complex64)
        z = np.array([[1,0],[0,-1]], dtype=np.complex64)
        x = np.array([[0,1],[1,0]], dtype=np.complex64)
        y = np.array([[0,-1j],[1j,0]], dtype=np.complex64)
        
        zm=np.array([0,1], dtype=np.complex64)
        xm=np.array([1,-1], dtype=np.complex64)/np.sqrt(2)
        ym=np.array([1,-1j], dtype=np.complex64)/np.sqrt(2)
        
        zp=np.array([1,0], dtype=np.complex64)
        xp=np.array([1,1], dtype=np.complex64)/np.sqrt(2)
        yp=np.array([1,1j], dtype=np.complex64)/np.sqrt(2)
        
        sigVec=np.array([eye,x,y,z])
    
        clearLog(nqubits)
        
        with open('.//logs//MEMLOG'+str(nqubits)+'.txt', 'w') as f:
            f.write('start')
            
        memprocess = multiprocessing.Process(target=mem_monitor_thr, args=(memtimeout,nqubits)) 
        memprocess.daemon=True
        memprocess.start()
        
    
    
    
        startHt = time.time()
        Heff=sp.csr_matrix((dim,dim),dtype=np.complex64)
        #Heff=np.zeros((dim,dim), dtype=np.complex64)
    
            
                
        for ii in range(nqubits-2):
            if ii==3:
                Jmod=1+random_frac
            else:
                Jmod=1
                
            
    
            
            # Heff=Heff  +((3+rnd.random())/4)*Jmod*J *( Delta*(pauli_kron_sp(z,nqubits,[ii,ii+1]))  + (pauli_kron_sp(x,nqubits,[ii,ii+1]))  + (pauli_kron_sp(y,nqubits,[ii,ii+1])))  +((3+rnd.random())/4)*Jmod*(J) *((Delta*pauli_kron_sp(z,nqubits,[ii,ii+2]))  + (pauli_kron_sp(x,nqubits,[ii,ii+2]))  + (pauli_kron_sp(y,nqubits,[ii,ii+2])))  
            Heff=Heff  +(1)*Jmod*J *( Delta*(pauli_kron_sp(z,nqubits,[ii,ii+1]))  + (pauli_kron_sp(x,nqubits,[ii,ii+1]))  + (pauli_kron_sp(y,nqubits,[ii,ii+1])))  +(0.5)*Jmod*(J) *((Delta*pauli_kron_sp(z,nqubits,[ii,ii+2]))  + (pauli_kron_sp(x,nqubits,[ii,ii+2]))  + (pauli_kron_sp(y,nqubits,[ii,ii+2])))  
                  
        
            
        # add the last pair
        Heff=Heff  +(1)*J *( Delta*(pauli_kron_sp(z,nqubits,[nqubits-2,nqubits-1]))  + (pauli_kron_sp(x,nqubits,[nqubits-2,nqubits-1]))  + (pauli_kron_sp(y,nqubits,[nqubits-2,nqubits-1])))
            
        stopHt = time.time()
        print('Done H in t=' + str((stopHt-startHt)/60))
        writeLog('Done H in t=' + str((stopHt-startHt)/60),nqubits)
        
        
        
        enrgy_val_block=np.zeros(dim)
        S2_val_block=np.zeros(dim)
        # m_val_block=[]
        
        
        #enrgy_vec_block=np.zeros([dim,dim])
        
        
        blockcount=0
        
        
        blockcounts=[]
        blocklens=[]
        
        Heff_block=sp.csr_matrix((dim,dim))
        
        maxS=nqubits//2
        # targets=[]
        # for i in range(0,maxS+1):
        #     for j in range(0, 2*i+1):
        #         targets.append([i,-i+j])
    
        targetsS=np.arange(0,maxS+1)
       
        E_S2_m_block=np.zeros([dim,3])
           
        E_S2_m_block_test=np.zeros([dim,1])
                            
        for targetS in targetsS:
            
            blockcountStartS=blockcount
        
            startSt=time.time()
            
            # enrgy_vec_S_total=[]
           
            
           
            Morder=[]
            
            targetMs=np.arange(-targetS,targetS+1)
            
            for m in targetMs:
                psisoutS=[]
                if m>=0:
                    with open('..//statesSM//StatesSplitwork_Q'+str(nqubits)+'_S='+str(targetS)+'_M='+str(m)+'.pickle', 'rb') as f:
                        psisout=pickle.load(f)
                    for psi in psisout:
                        psisoutS.append(psi)
                        Morder.append(m)
                else:
                    with open('..//statesSM//StatesSplitwork_Q'+str(nqubits)+'_S='+str(targetS)+'_M='+str(-m)+'.pickle', 'rb') as f:
                        psisout=pickle.load(f)
                    for psi in psisout:
                        psisoutS.append(sp.bsr_matrix(-psi.toarray()[0][::-1]))
                        Morder.append(m)

                    # m_val_block.append(m)
                
                Heff_S=np.zeros([len(psisoutS),len(psisoutS)])
                
                Us=np.zeros([dim,len(psisoutS)])
                
                for i in range(0, len(psisoutS)):
                    Us[:,i] = psisoutS[i].toarray()
                    
                Heff_S=np.conjugate(np.transpose(Us))@Heff@Us
                
                # for i in range(0, len(psisoutS)):
                #     for j in range(0, len(psisoutS)):
                #         if j>=i:
                #             Heff_S[i,j]=np.array(np.conjugate((np.reshape(psisoutS[j].toarray(),[1,dim])))@Heff@np.reshape(psisoutS[i].toarray(),[dim,1]))[0][0]
                #             Heff_S[j,i]=Heff_S[i,j]
                            
                startEsolveStime=time.time()
                
                enrgy_val_S, enrgy_vec_S=LA.eigh(Heff_S)
                
                del Heff_S
                
                stopEsolveStime=time.time()#
                
                print('Done SEsolve='+str(targetS)+' in t=' + str((stopEsolveStime-startEsolveStime)/60))
                writeLog('Done SEsolve='+str(targetS)+' in t=' + str((stopEsolveStime-startEsolveStime)/60),nqubits)
                
                enrgy_vec_M_total=(sp.csc_matrix(Us@enrgy_vec_S))
                #enrgy_vec_block[:,blockcount:blockcount+len(enrgy_val_S)]=Us@enrgy_vec_S
                # enrgy_vec_S=enrgy_vec_S.astype(np.float32)
                # enrgy_vec_S=sp.csc_matrix(enrgy_vec_S)
                del enrgy_vec_S
                del Us
                
                # for kkk in range(0,len(enrgy_val_S)):
                enrgy_val_block[blockcount:blockcount+len(enrgy_val_S)]=enrgy_val_S
                S2_val_block[blockcount:blockcount+len(enrgy_val_S)]=targetS*(targetS+1)
                
                E_S2_m_block[blockcount:blockcount+len(enrgy_val_S),1]=S2_val_block[blockcount:blockcount+len(enrgy_val_S)]
                E_S2_m_block[blockcount:blockcount+len(enrgy_val_S),0]=enrgy_val_block[blockcount:blockcount+len(enrgy_val_S)]
                
                #z_tot2=np.conjugate(np.transpose(Us))@z_tot@Us
                E_S2_m_block[blockcount:blockcount+len(enrgy_val_S),2]=m
                
                # enrgy_vec_S_to_E=Us@enrgy_vec_S[:,kkk]#??????????????????????????
                
                #enrgy_vec_block[:,blockcount]=np.reshape(enrgy_vec_S_to_E,[dim])
                
                print(blockcount)
                writeLog(str(blockcount),nqubits)
                
                blockcounts.append(blockcount)
                
                blocklens.append(len(enrgy_val_S))
                blockcount=blockcount+len(enrgy_val_S)
                
                with open('.//pickles//'+str(nqubits)+'Q_HeisBlockD_S='+str(targetS)+'M='+str(m)+'splitMSavebyM.pickle', 'wb') as f:
                    pickle.dump(enrgy_vec_M_total.astype(np.float32),f)
                
                    
                stopSt=time.time()
                print('Done S='+str(targetS)+'M='+str(m)+ ' in t=' + str((stopSt-startSt)/60))
                writeLog('Done S='+str(targetS)+'M='+str(m)+ ' in t=' + str((stopSt-startSt)/60),nqubits)
                
              
            
            
            #WRITE
            # Esort=np.argsort(E_S2_m_block[5:32,0])
            # enrgy_val_sort=E_S2_m_block[5:32,0][Esort]
            # enrgy_vec_sort=enrgy_vec_S_total_arr[:,Esort]
            # with open('.//pickles//'+str(nqubits)+'Q_HeisBlockD_S='+str(targetS)+'M='+str(targetM)+'splitMSavebyM.pickle', 'wb') as f:
            #     pickle.dump(enrgy_vec_S_total,f)
                
           
                
            # z_tot_block=0.5*sp.csc_matrix(pauli_sum(z,int(np.log2(len(enrgy_val_S)))), dtype=np.float32)   
            
            # E_S2_m_block[blockcount:blockcount+len(enrgy_val_S),2] = round(np.real(np.conjugate(np.transpose(enrgy_vec_block[:,i]))@z_tot_block@enrgy_vec_block[:,i]))
        # vectorBlocks=[]
        
        # for s in range(nqubits//2 + 1):
        #     with open(str(nqubits)+'Q_HeisBlockD_S='+str(s)+'.pickle', 'rb') as f:
        #         enrgy_vec_S = pickle.load(f)
        #         enrgy_vec_S=enrgy_vec_S[0]
        #         vectorBlocks.append(enrgy_vec_S)
        # enrgy_vec_block=sp.block_diag(vectorBlocks)
        # del vectorBlocks
        # enrgy_vec_block=enrgy_vec_block.tocsc()
        
        # for i in range(0,dim):
        #     E_S2_m_block[i,2]=round(np.real(np.conjugate(np.transpose(enrgy_vec_block.getcol(i)))@z_tot@enrgy_vec_block.getcol(i)).toarray()[0][0])
        

        
            #blockcounts.append(blockcount)
        
        # for i in range(0,dim):
        #     E_S2_m_block[i,1]=S2_val_block[i]#round(np.real(np.conjugate(np.transpose(enrgy_vec_block[:,i]))@S2@enrgy_vec_block[:,i]))#S2_val_block[i]
        #     E_S2_m_block[i,0]=enrgy_val_block[i]#(np.real(np.conjugate(np.transpose(enrgy_vec_block[:,i]))@Heff@enrgy_vec_block[:,i]))#enrgy_val_block[i]
            #E_S2_m_block[i,2]=round(np.real(np.conjugate(np.transpose(enrgy_vec_block[:,i]))@z_tot@enrgy_vec_block[:,i]))
            
        #WRITE
        with open('.//pickles//'+str(nqubits)+'Q_EsolvedHeisBlockD_Delta'+str(Delta)+'splitMSavebyM.pickle', 'wb') as f:
            pickle.dump([enrgy_val_block, E_S2_m_block],f)
        
        with open('.//pickles//'+str(nqubits)+'Q_HeisBlockD_BlockcountsSavebyM.pickle', 'wb') as f:
            pickle.dump(blockcounts,f)
    
    TEND=time.time()
    print("Time elapsed:")
    print(TEND-TSTART)
    writeLog("Time elapsed:",nqubits)
    writeLog(str(TEND-TSTART),nqubits)
