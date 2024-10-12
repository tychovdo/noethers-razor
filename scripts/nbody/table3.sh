#!/bin/bash
cd ../..

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99


# HNN
python main.py --batch_size 20 --vi --measure_scale 1.0 --init_prior_var 1.0 --output_var 0.001 --dynamics nbody_2d_3b --train_size 200 --train_steps 50 --stepsize 1.0 --act_fun elu --n_hidden 250 --n_layers 4 --n_epochs 1000 --sym_samples 100 --sym_over_path --lr_param 0.001 --lr_hyper 0.1 --prior_prec_method empirical --basis_prec 1.0 --output_var_method fixed --init_posterior_scale 10.0 --output_noise 0.001 --lr_cyper 0.01 --mc_samples 2 --conserved_init_std 0.01 --n_burnin 0 --measure_dist ball --bch_order 1 --load_model *enter wandb warmup model*

# HNN + fixed SE2
python main.py --group quadratic_se2 --batch_size 20 --vi --measure_scale 1.0 --init_prior_var 1.0 --output_var 0.001 --dynamics nbody_2d_3b --train_size 200 --train_steps 50 --stepsize 1.0 --act_fun elu --n_hidden 250 --n_layers 4 --n_epochs 1000 --sym_samples 100 --sym_over_path --lr_param 0.001 --lr_hyper 0.1 --prior_prec_method empirical --basis_prec 1.0 --output_var_method fixed --init_posterior_scale 10.0 --output_noise 0.001 --lr_cyper 0.01 --mc_samples 2 --conserved_init_std 0.01 --n_burnin 0 --measure_dist ball --bch_order 1 --load_model *enter wandb warmup model*

# HNN + learned conserved quantities (used max: 10)
python main.py --group quadratic_learn10 --batch_size 20 --vi --measure_scale 1.0 --init_prior_var 1.0 --output_var 0.001 --dynamics nbody_2d_3b --train_size 200 --train_steps 50 --stepsize 1.0 --act_fun elu --n_hidden 250 --n_layers 4 --n_epochs 1000 --sym_samples 100 --sym_over_path --lr_param 0.001 --lr_hyper 0.1 --prior_prec_method empirical --basis_prec 1.0 --output_var_method fixed --init_posterior_scale 10.0 --output_noise 0.001 --lr_cyper 0.01 --mc_samples 2 --conserved_init_std 0.01 --n_burnin 0 --measure_dist ball --bch_order 1 --load_model *enter wandb warmup model*
-salad-349


