import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['font.serif'] = 'Helvetica'

matplotlib.rcParams['font.weight'] = 'bold'

matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.labelsize'] = 'large'

matplotlib.rcParams['legend.fontsize'] = 'large'

matplotlib.rcParams['axes.titleweight'] = 'bold'

matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'

matplotlib.rcParams['lines.markersize'] = 10.0

def square_l2_norm(vec):
    tmp = 0
    for i in range(len(vec)):
        tmp = tmp + vec[i]**2
    return tmp

def cal_w1w2(w1,w2):
    tmp = 0
    for i in range(len(w1)):
        tmp = tmp+ w1[i]*w2[i]
    return tmp

def w1w2theta(W):
    theta = cal_w1w2(W[0,:],W[1,:])/(np.sqrt(square_l2_norm(W[0,:]))*np.sqrt(square_l2_norm(W[1,:])))
    return theta

def cal_projection(W_list,d):
    w1 = W_list[len(W_list)-1][0,:]
    w2 = W_list[len(W_list)-1][1,:]
    A = np.zeros((d,2))
    for i in range(d):
        A[i][0] = w1[i]
        A[i][1] = w2[i]
    P_matrix =  np.dot(np.dot(A,np.linalg.inv(np.dot(A.T,A))),A.T)
    e1 = np.zeros(d)
    e2 = np.zeros(d)
    e1[0]=1
    e2[1]=1
    P_e1 = np.dot(P_matrix,e1)
    P_e2 = np.dot(P_matrix,e2)
    P_value_e1 = np.sqrt(square_l2_norm(P_e1))
    P_value_e2 = np.sqrt(square_l2_norm(P_e2))
    return P_value_e1,P_value_e2

def get_data(choose_seed,path,d):
    w11 = []
    w12 = []
    w21 = []
    w22 = []

    w11_nor = []
    w12_nor = []
    w21_nor = []
    w22_nor = []
    theta_list = []
    P_value_e1_list = []
    P_value_e2_list = []
    path_alg = []
    for i in range(len(choose_seed)):
        path_alg.append(path + 'W_SSL_condition_num_0_d_10_tau_7_seed_'+str(choose_seed[i])+'_times_4000_lr_0.001.npz' )
        ### You need to replace the above path with the path of the result file
        ### Examples (1) SSL results: 'W_SSL_condition_num_0_d_10_tau_7_seed_'+str(choose_seed[i])+'_times_4000_lr_0.001.npz' 
        ### Examples (2) SL results: 'W_SL_d_10_tau_7_seed_'+str(choose_seed[i])+'_times_8000_lr_0.001.npz' 
    for i in range(len(choose_seed)):
        data =  np.load(path_alg[i])
        W_list = data['arr_0']
        w11.append(W_list[len(W_list)-1][0][0])
        w12.append(W_list[len(W_list)-1][0][1])
        w21.append(W_list[len(W_list)-1][1][0])
        w22.append(W_list[len(W_list)-1][1][1])
        w11_nor.append(W_list[len(W_list)-1][0][0]/np.sqrt(square_l2_norm(W_list[len(W_list)-1][0,:])))
        w12_nor.append(W_list[len(W_list)-1][0][1]/np.sqrt(square_l2_norm(W_list[len(W_list)-1][0,:])))
        w21_nor.append(W_list[len(W_list)-1][1][0]/np.sqrt(square_l2_norm(W_list[len(W_list)-1][1,:])))
        w22_nor.append(W_list[len(W_list)-1][1][1]/np.sqrt(square_l2_norm(W_list[len(W_list)-1][1,:])))
        theta_list.append(w1w2theta(W_list[len(W_list)-1]))
        P_value_e1,P_value_e2 = cal_projection(W_list,d)
        P_value_e1_list.append(P_value_e1)
        P_value_e2_list.append(P_value_e2)
    return w11, w12, w21, w22, w11_nor,w12_nor,w21_nor,w22_nor,theta_list,P_value_e1_list,P_value_e2_list


d = 10 ### the space dimension
epoch = 20 ### the number of the random seed
choose_seed = np.arange(epoch)
path = 'The path of the SSL/SL results' ### You need to change the path to the path where you saved the SSL/SL result
w11, w12, w21, w22, w11_nor,w12_nor,w21_nor,w22_nor,theta_list,P_value_e1_list,P_value_e2_list = get_data(choose_seed,path,d)

sns.set_theme(style="whitegrid")
sns.histplot(data=P_value_e1_list,bins=8)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('Frequency',fontsize=30)
plt.ylabel('The projection of ' r'$ e_1$',fontsize=30)

plt.show()
