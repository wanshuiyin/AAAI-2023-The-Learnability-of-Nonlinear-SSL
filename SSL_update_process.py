import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

## 初始化
def initialization(d,condition_num,tau):
    W = np.zeros((2,d))
    if condition_num == 0:  
        for i in range(2):
            for j in range(d):
                tmp = np.random.normal(0,1)
                W[i][j] = tmp
        W[0][0]  =random.uniform(3.1,3.9)

        W[1][0]  =-random.uniform(3.1,3.9)

        W[0][1]  =random.uniform(8.5,9)/tau

        W[1][1]  =random.uniform(8.5,9)/tau
    if condition_num == 1:  ###### only the correct sign
        for i in range(2):
            for j in range(d):
                tmp = np.random.normal(0,1)
                W[i][j] = tmp
        if W[0][0] <0: ## w11 positive 
            tmp = W[0][0]
            W[0][0]  =-tmp
        if W[1][0] >0: ## w21 negative
            tmp = W[1][0]
            W[1][0]  =-tmp
        if W[0][1]< 0: ## w12 positive 
            tmp = W[0][1]
            W[0][1]  =-tmp
        if W[1][1]<0: ## w22 positive 
            tmp = W[1][1]
            W[1][1]  =-tmp

    return W


def sigmoid(x):
	return 1/(1+np.exp(-x))

def dsigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def datapoint(n,d,rho,tau):
    e1 = np.zeros(d)
    e2 = np.zeros(d)
    e1[0] =1 
    e2[1] =1

    class_1 = []
    class_2 = []
    x = np.array([1, 2, 3, 4])
    prob = np.array([0.25, 0.25, 0.25,0.25])
    y  = np.random.choice(a=x, size=n, replace=True, p=prob)
    for i in range(n):
        if y[i] == 1:
            tmp_x11= e1 + rho*np.random.normal(0,1)
            class_1.append(tmp_x11)
        if y[i] == 2:
            tmp_x12= e1 +tau*e2+ rho*np.random.normal(0,1)
            class_1.append(tmp_x12)
        if y[i] == 3:
            tmp_x21= -e1 + rho*np.random.normal(0,1)
            class_2.append(tmp_x21)
        if y[i] == 4:
            tmp_x22= -e1 +tau*e2+ rho*np.random.normal(0,1)
            class_2.append(tmp_x22)
    datapoint_list = class_1+class_2
    return datapoint_list

def data_aug(x,d,rho):
    tmp = np.zeros((2,d))
    for i in range(d):
        tmp[0,i] = x[i] +rho*np.random.normal(0,1)
        tmp[1,i] = x[i] +rho*np.random.normal(0,1)
    return tmp

def d_sigmod(W,x,d,rho):
    aug_x = data_aug(x,d,rho)
    d_w1 = sigmoid(np.dot(W[0,:],aug_x[0,:]))*(1-sigmoid(np.dot(W[0,:],aug_x[0,:])))*sigmoid(np.dot(W[0,:],aug_x[1,:]))*aug_x[0,:] + sigmoid(np.dot(W[0,:],aug_x[1,:]))*(1-sigmoid(np.dot(W[0,:],aug_x[1,:])))*sigmoid(np.dot(W[0,:],aug_x[0,:]))*aug_x[1,:]
    d_w2 = sigmoid(np.dot(W[1,:],aug_x[0,:]))*(1-sigmoid(np.dot(W[1,:],aug_x[0,:])))*sigmoid(np.dot(W[1,:],aug_x[1,:]))*aug_x[0,:] + sigmoid(np.dot(W[1,:],aug_x[1,:]))*(1-sigmoid(np.dot(W[1,:],aug_x[1,:])))*sigmoid(np.dot(W[1,:],aug_x[0,:]))*aug_x[1,:]
    return d_w1,d_w2

def d_regular(W,alpha,n): 
    d_w1 = -(2*W[0,:])*alpha*n
    d_w2 = -(2*W[1,:])*alpha*n
    return d_w1,d_w2

def update_W(W,eta,datapoint_list,d,alpha,rho,n):
    gradient_w1,gradient_w2 = d_regular(W,alpha,n) 
    for i in range(len(datapoint_list)):
        tmp_gradient_w1,tmp_gradient_w2 = d_sigmod(W,datapoint_list[i],d,rho)
        gradient_w1 = gradient_w1+tmp_gradient_w1
        gradient_w2 = gradient_w2+tmp_gradient_w2
        tmp_gradient_w1 = np.zeros(d)
        tmp_gradient_w2 = np.zeros(d)
    W_tmp = np.zeros((2,d))
    W_tmp[0,:] = W[0,:] + eta*gradient_w1
    W_tmp[1,:] = W[1,:] + eta*gradient_w2
    return W_tmp

def update_process(W,eta,T,n,d,datapoint_list,alpha,rho):
    W_list = []
    W_list.append(W)
    for i in tqdm(range(T)):
        W_tmp = update_W(W_list[len(W_list)-1],eta,datapoint_list,d,alpha,rho,n)
        W_list.append(W_tmp)
    return W_list

def run(epoch,eta,T,n,d,alpha,rho,tau,condition,path):
    for i in tqdm(range(epoch)):
        np.random.seed(i)
        random.seed(i)
        W= initialization(d,condition,tau)
        datapoint_list = datapoint(n,d,rho,tau)
        W_list= update_process(W,eta,T,n,d,datapoint_list,alpha,rho)
        np.savez(path+'W_SSL_condition_num_'+str(condition)+'_d_'+str(d)+'_tau_'+str(tau)+'_seed_'+str(i)+'_times_'+str(T)+'_lr_'+str(eta),W_list)

d = 10  ### the space dimension
n = d**2 ### the number of datapoints
T = 4000 ### the number of the iteration
eta = 0.001  ### the learning rate
alpha = 1/800 ## the coefficient of W
rho = 1/(d**(1.5))  ###  the coefficient of datapoints and data augmentation noise terms
tau = 7  ### the coefficient of e_2
epoch = 20 ### the number of the random seed
condition = 0 ### initialization strategy:  (1) condtion =0 initialization around local minimum (2)condtion =0 only the correct sign
path = 'The path of the SSL results' ### You need to change the path to the path where you saved the SSL result
run(epoch,eta,T,n,d,alpha,rho,tau,condition)
