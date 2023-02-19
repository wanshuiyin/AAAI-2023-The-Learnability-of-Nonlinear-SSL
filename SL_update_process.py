
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

## 初始化
def initialization_W_SL(d):
    W_SL = np.zeros((2,d)) 
    for i in range(2):
        for j in range(d):
            tmp = np.random.normal(0,1)
            W_SL[i][j] = tmp
    if W_SL[0][0] <0: ## w11大于0
        tmp = W_SL[0][0]
        W_SL[0][0]  =-tmp
    if W_SL[1][0] >0: ## w21小于0
        tmp = W_SL[1][0]
        W_SL[1][0]  =-tmp
    return W_SL

def initialization_F():
    F = np.zeros(2)
    for i in range(2):
        tmp = np.random.normal(0,1)
        F[i] = tmp
    return F

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
    class_1_label = []
    class_2_label = []
    x = np.array([1, 2, 3, 4])
    prob = np.array([0.25, 0.25, 0.25,0.25])
    y  = np.random.choice(a=x, size=n, replace=True, p=prob)
    for i in range(n):
        if y[i] == 1:
            tmp_x11= e1 + rho*np.random.normal(0,1)
            class_1.append(tmp_x11)
            class_1_label.append(0)
        if y[i] == 2:
            tmp_x12= e1 +tau*e2+ rho*np.random.normal(0,1)
            class_1.append(tmp_x12)
            class_1_label.append(0)
        if y[i] == 3:
            tmp_x21= -e1 + rho*np.random.normal(0,1)
            class_2.append(tmp_x21)
            class_2_label.append(1)
        if y[i] == 4:
            tmp_x22= -e1 +tau*e2+ rho*np.random.normal(0,1)
            class_2.append(tmp_x22)
            class_2_label.append(1)
    datapoint_list = class_1+class_2
    datapoint_label_list = class_1_label+class_2_label
    return datapoint_list,datapoint_label_list

def d_W_SL_each_point(F,W_SL,x,y):
    hat_y = sigmoid(F[0]*sigmoid(np.dot(W_SL[0,:],x))+F[1]*sigmoid(np.dot(W_SL[1,:],x)))
    d_w1 = (-y/hat_y+(1-y)/(1-hat_y))*hat_y*(1-hat_y)*sigmoid(np.dot(W_SL[0,:],x))*(1-sigmoid(np.dot(W_SL[0,:],x)))*F[0]*x
    d_w2 = (-y/hat_y+(1-y)/(1-hat_y))*hat_y*(1-hat_y)*sigmoid(np.dot(W_SL[1,:],x))*(1-sigmoid(np.dot(W_SL[1,:],x)))*F[1]*x
    return d_w1, d_w2

def d_F_each_point(F,W_SL,x,y):
    hat_y = sigmoid(F[0]*sigmoid(np.dot(W_SL[0,:],x))+F[1]*sigmoid(np.dot(W_SL[1,:],x)))
    d_f1 = (-y/hat_y+(1-y)/(1-hat_y))*hat_y*(1-hat_y)*sigmoid(np.dot(W_SL[0,:],x))
    d_f2 = (-y/hat_y+(1-y)/(1-hat_y))*hat_y*(1-hat_y)*sigmoid(np.dot(W_SL[1,:],x))
    return d_f1, d_f2

def d_W_SL_regular(W_SL,beta_SL,n):  
    d_w1 = (2*W_SL[0,:])*beta_SL*n
    d_w2 = (2*W_SL[1,:])*beta_SL*n
    return d_w1,d_w2

def d_F_regular(F,gamma_F,n):  
    d_f1 = (2*F[0])*gamma_F*n
    d_f2 = (2*F[1])*gamma_F*n
    return d_f1,d_f2

def update_F(F,W_SL,eta,datapoint_list,datapoint_label_list,gamma_F,n):
    gradient_f1,gradient_f2 = d_F_regular(F,gamma_F,n) 
    for i in range(len(datapoint_list)):
        tmp_gradient_f1,tmp_gradient_f2 = d_F_each_point(F,W_SL,datapoint_list[i],datapoint_label_list[i])
        gradient_f1 = gradient_f1+tmp_gradient_f1
        gradient_f2 = gradient_f2+tmp_gradient_f2
        tmp_gradient_f1 = 0
        tmp_gradient_f2 = 0
    F_tmp = np.zeros(2)
    F_tmp[0] = F[0] - eta*gradient_f1  ### gradient descent
    F_tmp[1] = F[1] - eta*gradient_f2
    return F_tmp

def update_W_SL(F,W_SL,eta,datapoint_list,datapoint_label_list,beta_SL,d,n):
    gradient_w1,gradient_w2 = d_W_SL_regular(W_SL,beta_SL,n) 
    for i in range(len(datapoint_list)):
        tmp_gradient_w1,tmp_gradient_w2 = d_W_SL_each_point(F,W_SL,datapoint_list[i],datapoint_label_list[i])
        gradient_w1 = gradient_w1+tmp_gradient_w1
        gradient_w2 = gradient_w2+tmp_gradient_w2
        tmp_gradient_w1 = 0
        tmp_gradient_w2 = 0
    W_tmp = np.zeros((2,d))
    W_tmp[0,:] = W_SL[0,:] - eta*gradient_w1 ### gradient descent
    W_tmp[1,:] = W_SL[1,:] - eta*gradient_w2
    return W_tmp

def update_process(F,W_SL,eta,T,d,datapoint_list,datapoint_label_list,gamma_F,beta_SL,n):
    W_list = []
    W_list.append(W_SL)
    F_list = []
    F_list.append(F)
    for i in tqdm(range(T)):
        W_tmp = update_W_SL(F_list[len(F_list)-1],W_list[len(W_list)-1],eta,datapoint_list,datapoint_label_list,beta_SL,d,n)
        W_list.append(W_tmp)
        F_tmp = update_F(F_list[len(F_list)-1],W_list[len(W_list)-1],eta,datapoint_list,datapoint_label_list,gamma_F,n)
        F_list.append(F_tmp)
    return W_list, F_list

def run(epoch,eta,T,n,d,gamma_F,beta_SL,rho,tau,path):
    for i in tqdm(range(epoch)):
        np.random.seed(i)
        random.seed(i)
        W_SL= initialization_W_SL(d)
        F = initialization_F()
        datapoint_list,datapoint_label_list = datapoint(n,d,rho,tau)
        W_list, F_list= update_process(F,W_SL,eta,T,d,datapoint_list,datapoint_label_list,gamma_F,beta_SL,n)
        np.savez(path+'W_SL_d_'+str(d)+'_tau_'+str(tau)+'_seed_'+str(i)+'_times'+str(T)+'_lr_'+str(eta),W_list)
        np.savez(path+'F_d_'+str(d)+'_tau_'+str(tau)+'_seed_'+str(i)+'_times_'+str(T)+'_lr_'+str(eta),F_list)


d = 10  ### the space dimension
n = d**2 ### the number of datapoints
T = 8000 ### the number of the iteration
eta = 0.001  ### the learning rate
beta_SL = 1/800 ## the coefficient of W^{\text{SL}}
gamma_F = 1/800 ## the coefficient of F
rho = 1/(d**(1.5))  ###  the coefficient of datapoints and data augmentation noise terms
tau = 7  ### the coefficient of e_2
epoch = 20 ### the number of the random seed
path = 'The path of the SL results' ### You need to change the path to the path where you saved the SL result



