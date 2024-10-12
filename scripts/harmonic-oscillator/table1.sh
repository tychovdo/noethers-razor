#!/bin/bash
cd ../..

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99

python main.py --group quadratic_so2 --train_size 7 --train_steps 4 --steps 20 --stepsize 0.20 --sym_steps 50 --vi --measure_scale 1.0 --init_prior_var 1 --output_var 0.001 --output_noise 0.001 --dynamics harmonic-oscillator --act_fun elu2 --n_layers 3 --n_hidden 200 --sym_samples 200 --sym_over_path --lr_param 0.001 --mc_samples 20 --n_epochs 5000 --fixed_basis_scale 1.0 --seed 105
python main.py --group quadratic_learn1 --train_size 7 --train_steps 4 --steps 20 --stepsize 0.20 --sym_steps 50 --vi --measure_scale 1.0 --init_prior_var 1 --output_var 0.001 --output_noise 0.001 --dynamics harmonic-oscillator --act_fun elu2 --n_layers 3 --n_hidden 200 --sym_samples 200 --sym_over_path --lr_param 0.001 --mc_samples 20 --n_epochs 5000 --fixed_basis_scale 1.0 --seed 105
python main.py --train_size 7 --train_steps 4 --steps 20 --stepsize 0.20 --sym_steps 50 --vi --measure_scale 1.0 --init_prior_var 1 --output_var 0.001 --output_noise 0.001 --dynamics harmonic-oscillator --act_fun elu2 --n_layers 3 --n_hidden 200 --sym_samples 200 --sym_over_path --lr_param 0.001 --mc_samples 20 --n_epochs 5000 --fixed_basis_scale 1.0 --seed 105

