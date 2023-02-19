# AAAI-2023-The-Learnability-of-Nonlinear-SSL
This project is the code for AAAI 2023 paper "Understanding Representation Learnability of Nonlinear Self-Supervised Learning".

# Code of  Simulation Experiments

We conduct simulation experiments on the nonlinear SSL model and the nonlinear SL model, and their update processes are in the following files:

- SSL_update_process.py
- SL_update_process.py

We will illustrate the role of hyperparameters. The hyperparameters d, n, T, $\eta$, $\alpha$, $\rho$, $\tau$ in the program have the same meaning as the hyperparameters in main paper. The hyperparameters $\beta_SL$ and $\gamma_F$ in SL_update_process.py are $\beta$ and $\gamma$ in the main paper. The hyperparameters epoch in the program is the number of the random seeds. The hyperparameters condition in the program is used to determine to determine which initialization strategy is used.  When condition=0, we initialize $w_1$ and $w_2$ around the local minimum. When condition=1, we only guarantee the correct sign. In the SL model, the hyperparameter condition does not exist.

## How to Run

You can modify the above hyperparameters to  get different results for the SSL or SL model. The only thing worth noting is that you need to change the hyperparameter path to the path where the result is stored when running the program.

## How to Plot Figure

In the folder, we include four programs:

- Figure_learning_curve.py
- Figure_learning_results.py
- Figure_projection_e1.py
- Figure_projection_e2.py

, and each program contains a function "get_data". You only need to change the results name in this function to get the final figure. 
