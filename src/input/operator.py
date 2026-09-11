#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:27:37 2025

@author: ALasek
"""

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp

class operator:
    

    #this is called on creation of instance
    def __init__(self,  params,isreal):
        
        self.isreal=isreal
        dtypepbits = params["dtypepbits"]
        if dtypepbits == 32:
            self.dtypef = cp.float32
            self.dtypec = cp.complex64
        elif dtypepbits == 64:
            self.dtypef = cp.float64
            self.dtypec = cp.complex128
        else:
            raise ValueError(f"Unsupported dtypepbits {dtypepbits!r}")
        self.nqubitsS=params["nqubits_S"]
        self.nqubitsE=params["nqubits_E"]
        self.nqubits=self.nqubitsS+self.nqubitsE
        self.psi_S_cpspec=params["psi_S_spec"]
        self.psi_E_cpspec=params["psi_E_spec"]
        
        dim=2**self.nqubits
        
     
        if isreal:
            self.op_cpsp=cpsp.csr_matrix((dim, dim), dtype=self.dtypef)
        else:
            self.op_cpsp=cpsp.csr_matrix((dim, dim), dtype=self.dtypec)
        
        
        
  
    
    def pauli_sum_cpsp(self,sigma):
        n=self.nqubits
        #works only for 5 or more qubits
        self.op_cpsp=cpsp.csr_matrix((2**n,2**n),dtype=self.dtypec)
        #arr=[np.array([[1,0],[0,1]], dtype=np.complex64)]*n
        for ll in range(n):
            #arr[ll]=sigma
            #temp=cpsp.csr_matrix(arr[0])
            if ll==0:
                temp=cpsp.csr_matrix(sigma)
            else:
                temp=cpsp.csr_matrix(np.eye(2))
                
            for kk in range(1,n):
                if kk==ll:
                    temp=cpsp.kron(temp,cpsp.csr_matrix(sigma))
                else:
                    temp=cpsp.kron(temp,cpsp.csr_matrix(np.eye(2)))
            self.op_cpsp+=temp

    
    
  
 
    
    def pauli_kron_cpsp(self,sigma,pair):
        n=self.nqubits
        sigma=cpsp.bsr_matrix(sigma)
        i1=pair[0]
        i2=pair[1]
        
        if abs(i2-i1)>1:
            eye1=cpsp.eye(2**(n-i2-1),dtype=np.int8)
            eye2=cpsp.eye(2**(i2-i1-1),dtype=np.int8)
            eye3=cpsp.eye(2**(i1),dtype=np.int8)
            self.op_cpsp+= cpsp.csr_matrix(cpsp.kron(eye3,cpsp.kron(sigma,cpsp.kron(eye2,cpsp.kron(sigma,eye1)))))
        else:
            eye1=cpsp.eye(2**(n-i2-1),dtype=np.int8)
            eye3=cpsp.eye(2**(i2-1),dtype=np.int8)
            self.op_cpsp+= cpsp.csr_matrix(cpsp.kron(eye3,cpsp.kron(sigma,cpsp.kron(sigma,eye1))))
    
    def pauli_single(self,sigma,ind):
        
        n=self.nqubits
        
        arr=[cp.array([[1,0],[0,1]], dtype=self.dtypec)]*n
        arr[ind]=sigma
        temp=arr[0]
        for kk in range(1,n):
            temp=cp.kron(temp,arr[kk])
        
        self.op_cpsp+=cpsp.csr_matrix(temp)

    def pauli_single_cpsp(self,sigma,ind):
        
        n=self.nqubits
        
        eye=cpsp.csr_matrix(cp.array([[1,0],[0,1]],dtype=self.dtypec))
        if ind==0:
            temp=cpsp.csr_matrix(sigma)
        else:
            temp=eye
        for kk in range(1,n):
            if kk==ind:
                arr=cpsp.csr_matrix(sigma)
            else:
                arr=eye
            temp=cp.kron(temp,arr)
        
        self.op_cpsp+=cpsp.csr_matrix(temp)
