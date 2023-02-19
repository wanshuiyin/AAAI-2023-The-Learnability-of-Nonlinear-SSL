import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
plt.rcParams.update(params)
matplotlib.rcParams['font.size'] = 50
matplotlib.rcParams['font.serif'] = 'Helvetica'

matplotlib.rcParams['font.weight'] = 'bold'

matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.labelsize'] = 'large'

matplotlib.rcParams['legend.fontsize'] = 'large'

matplotlib.rcParams['axes.titleweight'] = 'bold'

matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'

matplotlib.rcParams['lines.markersize'] = 200.0

def get_data(choose_seed,path,T):
    all_w11 = np.zeros((choose_seed,T))
    all_w12 = np.zeros((choose_seed,T))
    all_w21 = np.zeros((choose_seed,T))
    all_w22 = np.zeros((choose_seed,T))
    path_alg = []
    for i in range(choose_seed):
        path_alg.append(path + 'W_SSL_condition_num_0_d_10_tau_7_seed_'+str(i)+'_times_4000_lr_0.001.npz' )
        ### You need to replace the above path with the path of the result file
        ### Examples (1) SSL results: 'W_SSL_condition_num_0_d_10_tau_7_seed_'+str(i)+'_times_4000_lr_0.001.npz' 
        ### Examples (2) SL results: 'W_SL_d_10_tau_7_seed_'+str(i)+'_times_8000_lr_0.001.npz' 
    for i in range(choose_seed):
        data =  np.load(path_alg[i])
        W_list = data['arr_0']
        for j in range(T):
            all_w11[i][j] = W_list[j][0][0]
            all_w12[i][j] = W_list[j][0][1]
            all_w21[i][j] = W_list[j][1][0]
            all_w22[i][j] = W_list[j][1][1]
    return all_w11, all_w12, all_w21, all_w22

def mean_std_plot(all_w11, all_w12, all_w21, all_w22):
    one_all_w11 = np.array([np.mean(all_w11[:, round_id]) for round_id in range(all_w11.shape[1])])
    std_all_w11 = np.array([np.std(all_w11[:, round_id]) for round_id in range(all_w11.shape[1])])
    one_all_w12 = np.array([np.mean(all_w12[:, round_id]) for round_id in range(all_w12.shape[1])])
    std_all_w12 = np.array([np.std(all_w12[:, round_id]) for round_id in range(all_w12.shape[1])])
    one_all_w21 = np.array([np.mean(all_w21[:, round_id]) for round_id in range(all_w21.shape[1])])
    std_all_w21 = np.array([np.std(all_w21[:, round_id]) for round_id in range(all_w21.shape[1])])
    one_all_w22 = np.array([np.mean(all_w22[:, round_id]) for round_id in range(all_w22.shape[1])])
    std_all_w22 = np.array([np.std(all_w22[:, round_id]) for round_id in range(all_w22.shape[1])])

    return one_all_w11, std_all_w11, one_all_w12, std_all_w12,one_all_w21, std_all_w21,one_all_w22, std_all_w22



d = 10 ### the space dimension
T = 8001 ### the number of the SSL/SL experiments iteration
chooos_seed =20 ### the number of the random seed
path = 'The path of the SSL/SL results' ### You need to change the path to the path where you saved the SSL/SL result
all_w11, all_w12, all_w21, all_w22 = get_data(chooos_seed,path,T)
one_all_w11, std_all_w11, one_all_w12, std_all_w12,one_all_w21, std_all_w21,one_all_w22, std_all_w22 = mean_std_plot(all_w11, all_w12, all_w21, all_w22)
x = np.arange(T)

sns.set_theme(style="whitegrid")
plt.figure()
plt.plot(x,one_all_w11,c ='blue',label = r'$\widetilde{w}_{1}^{\operatorname{SL}(1)}$',linewidth=3.0,marker= "v", markevery=2000,markersize=10)
plt.fill_between(x, one_all_w11 - 1.96 * std_all_w11/np.sqrt(chooos_seed),
                    one_all_w11 + 1.96 * std_all_w11/np.sqrt(chooos_seed), color='blue', alpha=0.1)
plt.plot(x,one_all_w12,c ='skyblue',label = r'$\widetilde{w}_{1}^{\operatorname{SL}(2)}$',linewidth=3.0,marker= "*", markevery=2000,markersize=10)
plt.fill_between(x, one_all_w12 - 1.96 * std_all_w12/np.sqrt(chooos_seed),
                    one_all_w12 + 1.96 * std_all_w12/np.sqrt(chooos_seed), color='skyblue', alpha=0.1)
plt.plot(x,one_all_w21,c ='red',label = r'$\widetilde{w}_{2}^{\operatorname{SL}(1)}$',linewidth=3.0,marker= "o", markevery=2000,markersize=10)
plt.fill_between(x, one_all_w21 - 1.96 * std_all_w21/np.sqrt(chooos_seed),
                    one_all_w21 + 1.96 * std_all_w21/np.sqrt(chooos_seed), color='red', alpha=0.1)
plt.plot(x,one_all_w22,c ='deeppink',label = r'$\widetilde{w}_{2}^{\operatorname{SL}(2)}$',linewidth=3.0,marker= "s", markevery=2000,markersize=10)
plt.fill_between(x, one_all_w22 - 1.96 * std_all_w22/np.sqrt(chooos_seed),
                    one_all_w22 + 1.96 * std_all_w22/np.sqrt(chooos_seed), color='deeppink', alpha=0.1)
plt.xticks(range(0,8001,2000),fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("Round T",fontsize=30)
plt.ylabel("Value",fontsize=30)
plt.legend(fontsize=25)
plt.show()

