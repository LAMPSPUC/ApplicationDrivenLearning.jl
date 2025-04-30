T_train = 300
T_test = 100
p = 4
r = 4
hidden_layers_size = 16
pretrain_time_limit = 60*10
time_limit = 60*1
compute_every = 10
max_iter = 10_000
max_iter_pretrain = 300_000
lr = 1e-3
batch_size = -1
max_params_nelder_mead = 400
max_I_jump_ls = 10
max_pretrain_tries = 3

I_space = [2]  # [1,2,10,20,50,100]
n_hidden_layers_space = [0,1]  # [0,1,2]
