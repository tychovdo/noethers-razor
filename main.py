from functools import partial
from pathlib import Path
import argparse
import numpy as np
import re

import copy

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import jax
import jax.numpy as jnp

import wandb

from dynamics import Nbody, HarmonicOscillator, NHarmonicOscillator

from data import generate_dataset 
from network import init_mlp, init_mlp_vi, init_mlp_vi_residual
from train import train_model
from save import save_model

def init_dynamics(dynamics_name):
    if 'nbody' in dynamics_name:
        _, dstr, bstr = dynamics_name.split('_')
        assert dstr[-1] == 'd', f"For nbody dynamics, use nbody_*d_*b format to specify dimensions and amount of bodies"
        dim = int(dstr[:-1])
        assert bstr[-1] == 'b', f"For nbody dynamics, use nbody_*d_*b format to specify dimensions and amount of bodies"
        n_bodies = int(bstr[:-1])

        dynamics = Nbody(dim, n_bodies)
    elif dynamics_name == 'harmonic-oscillator':
        dynamics = HarmonicOscillator()
    elif 'nharm' in dynamics_name:
        _, nstr = dynamics_name.split('_')
        n = int(nstr)
        dynamics = NHarmonicOscillator(n)
    else:
        raise NotImplementedError(f"Unknown dynamics: {dynamics}")
    
    return dynamics


def main(args):
    wandb.init(config=args, project='noether')

    # Seed
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Prepare data 
    dynamics = init_dynamics(args.dynamics)

    save_name = f"{args.dynamics}_train"
    initial_phase = partial(dynamics.initial_phase, q_scale=args.q_scale, p_scale=args.p_scale, q_shift=args.q_shift, p_shift=args.p_shift)

    train_trajectories = generate_dataset(dynamics.H, initial_phase, save_name=save_name,
                                          n_trajectories=args.train_size, n_steps=args.train_steps,
                                          stepsize=args.stepsize, seed=args.seed)

    train_dataset = TensorDataset(torch.tensor(train_trajectories.tolist()))

    def jax_collate(x):
        arr = jnp.array(x)
        return arr.reshape(arr.shape[0], arr.shape[2], arr.shape[3])

    if args.batch_size == -1:
        batch_size = len(train_dataset)
    else:
        batch_size = args.batch_size
    print("Batch size: ", batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=jax_collate
    ) 
    train_loader_noshuffle = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=jax_collate
    ) 
    print(f'train: {train_trajectories.shape}')

    test_loaders = {}

    for name, (qscale, pscale, qtrans) in [('test', (1.0, 0.2, 0.0)),
                                           ('test_wider', (3.0, 0.2, 0.0)),
                                           ('test_moved', (1.0, 0.2, 1.0))
                                           ]:

        save_name = f"{args.dynamics}_{name}"
        initial_phase = partial(dynamics.initial_phase, q_scale=qscale, p_scale=pscale, q_trans=qtrans)

        test_trajectories = generate_dataset(dynamics.H, initial_phase, save_name=save_name, n_trajectories=args.test_size, n_steps=args.test_steps, stepsize=args.stepsize, seed=args.seed)

        test_dataset = TensorDataset(torch.tensor(test_trajectories.tolist()))

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_size,
            shuffle=False,
            collate_fn=jax_collate
        )

        test_loaders[name] = test_loader

        print(f'Loaded dataset [{name}]: {test_trajectories.shape}')

    # Model
    key = jax.random.PRNGKey(100)

    if args.vi:
        if args.residual:
            params = init_mlp_vi_residual(key, dynamics.pdim, args.n_hidden, 1, args.n_layers, args.fixed_basis)
        else:
            params = init_mlp_vi(key, dynamics.pdim, args.n_hidden, 1, args.n_layers, args.fixed_basis, args.fixed_basis_scale)

        if args.basis_prec != 1.0:
            if args.fixed_basis:
                params[0]['fixed_mean'] = params[0]['fixed_mean'] / jnp.sqrt(args.basis_prec)
            else:
                params[0]['mean'] = params[0]['mean'] / jnp.sqrt(args.basis_prec)

        for l in range(len(params)):
            if 'S' in params[l]:
                params[l]['A'] = params[l]['A'] * args.init_posterior_scale / 5.0
                params[l]['S'] = params[l]['S'] * 1.0 / 5.0

        ps = [1 / args.init_prior_var for _ in range(args.n_layers)]
    else:
        params = init_mlp(key, dynamics.pdim, args.n_hidden, 1, args.n_layers, args.fixed_basis)

        ps = []
        for l in range(args.n_layers):
            p_l = 1 / args.init_prior_var
            ps.append(p_l)

    if 'mlp' in args.group:
        if 'learn' in args.group:
            # learned conserved quantity currently always uses a simple MLP, C(x) = MLP(x)
            group_dim = int(re.findall(r'\d+', args.group)[0])
            *subkeys, key = jax.random.split(key, group_dim + 1)
            cypers = [init_mlp(subkey, dynamics.pdim, args.n_hidden, 1, 2, False) for subkey in subkeys]
        else:   
            raise NotImplementedError(f"Unknown group. Got: {args.group}.")
    elif 'quadratic' in args.group:
        # quadratic conserved quantity: C(x) = <x, Wx>/2 + <b,x> + const., which yields linear closed-form solutions
        init_std = args.conserved_init_std
        if 'learn' in args.group:
            group_dim = int(re.findall(r'\d+', args.group)[0])
            *subkeys, key = jax.random.split(key, group_dim + 1)
            cypers = [dynamics.init_affine_as_subgroup('', key=subkey, std=init_std) for subkey in subkeys]

        elif 'tn' in args.group:
            assert isinstance(dynamics, Nbody), f"The group 'SE(n)' only implemented for Nbody dynamics."

            cypers = [dynamics.init_affine_as_subgroup(f't_{d}') for d in range(dynamics.dim)]

        elif 'so2' in args.group:
            cypers = [dynamics.init_affine_as_subgroup('rabs_0_1')]

        elif 'se2' in args.group:
            cypers = [dynamics.init_affine_as_subgroup('t_0'),
                      dynamics.init_affine_as_subgroup('t_1'),
                      dynamics.init_affine_as_subgroup('rabs_0_1')]

        elif 'sen' in args.group:
            assert isinstance(dynamics, Nbody), f"The group 'SE(n)' only implemented for Nbody dynamics."

            cypers = []
            for d1 in range(dynamics.dim):
                cypers.append(dynamics.init_affine_as_subgroup(f't_{d1}'))
                cypers.append(dynamics.init_affine_as_subgroup(f'p_{d1}'))

                for d2 in range(d1 + 1, dynamics.dim):
                    cypers.append(dynamics.init_affine_as_subgroup(f'rcom_{d1}_{d2}'))
                    cypers.append(dynamics.init_affine_as_subgroup(f'rabs_{d1}_{d2}'))
                    cypers.append(dynamics.init_affine_as_subgroup(f'q_{d1}_{d2}'))
        elif 'un'in args.group:
            group_dim = dynamics.cdim ** 2

            cypers = [dynamics.init_affine_as_subgroup(f'un_{i}') for i in range(group_dim)]
        else:
            raise NotImplementedError(f"Unknown group. Got: {args.group}.")
    elif args.group == '':
        cypers = []
    else:
        raise NotImplementedError(f"Unknown group. Got {args.group}")

    if 'learn' not in args.group:
        args.lr_cyper = 0.0

    output_var = args.output_var

    hypers = ps + [jnp.sqrt(output_var)]

    if args.load_model != "":
        print(f"Loading model: {args.load_model}")
        loaded_args = jnp.load(f'./checkpoints/{args.load_model}.args', allow_pickle=True)
        params, hypers, _ = jnp.load(f'./checkpoints/{args.load_model}.data', allow_pickle=True)

        for argname in ['n_hidden', 'n_layers', 'act_fun', 'fixed_basis', 'basis_prec']:
            loaded_arg = getattr(loaded_args, argname)
            current_arg = getattr(args, argname)
            assert loaded_arg == current_arg, f"Loaded args do not match args on [{argname}]. Got {loaded_arg}!={current_arg}."

    print('output_var:', output_var)

    ALLOW_MC_SYMMETRISATION = False
    if (not ALLOW_MC_SYMMETRISATION) and (not args.sym_over_path):
        print("[ERR] Only symmetrisation over paths is allowed. Either use --sym_over_path, or set ALLOW_MC_SYMMETRISATION=True.")
        exit(1)

    # Main train loop
    params, hypers, cypers = train_model(params, hypers, cypers, args.canonicalise, train_loader, test_loaders, train_loader_noshuffle,
                                         dynamics, args.group, args.n_epochs, args.n_layers, args.act_fun,
                                         steps=args.steps, stepsize=args.stepsize,
                                         scheduler=args.scheduler, lr_param=args.lr_param, lr_hyper=args.lr_hyper, lr_cyper=args.lr_cyper,
                                         n_burnin=args.n_burnin, cyper_burnin=args.cyper_burnin, vi=args.vi,
                                         measure_dist=args.measure_dist, measure_scale=args.measure_scale, sym_samples=args.sym_samples, sym_over_path=args.sym_over_path,
                                         map_mean=args.map_mean, stl=args.stl, output_var_method=args.output_var_method, output_noise=args.output_noise, prior_prec_method=args.prior_prec_method, sym_steps=args.sym_steps, bch_order=args.bch_order, mc_samples=args.mc_samples, tmp=args.tmp, beta1=args.beta1, beta2=args.beta2, residual=args.residual, curriculum=args.curriculum, multisteps=args.multisteps)


    # Save
    save_model((params, hypers, cypers), args, wandb.run.name)


if __name__=='__main__':
    parser = argparse.ArgumentParser("Training code")
    parser.add_argument('--dynamics', type=str, default='nbody_2d_3b') # dynamics
    parser.add_argument('--group', type=str, default='') # dynamics

    parser.add_argument('--seed', type=int, default=100)

    parser.add_argument('--train_size', type=int, default=100) 
    parser.add_argument('--train_steps', type=int, default=10)

    parser.add_argument('--test_size', type=int, default=100) 
    parser.add_argument('--test_steps', type=int, default=20)

    parser.add_argument('--stepsize', type=float, default=0.1)
    parser.add_argument('--n_conserved', type=int, default=3)

    parser.add_argument('--n_hidden', type=int, default=200) # architecture
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--act_fun', type=str, default='elu2')
    parser.add_argument('--fixed_basis', action='store_true')
    parser.add_argument('--fixed_basis_scale', type=float, default=1.0)
    parser.add_argument('--basis_prec', type=float, default=1.0)

    parser.add_argument('--n_epochs', type=int, default=1000) # training
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--cyper_burnin', type=int, default=0)
    parser.add_argument('--n_burnin', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--lr_param', type=float, default=0.001)
    parser.add_argument('--lr_hyper', type=float, default=0.1) 
    parser.add_argument('--lr_cyper', type=float, default=0.01) 
    parser.add_argument('--projection_method', type=str, default='svd')
    parser.add_argument('--canonicalise', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--output_var', type=float, default=0.001)
    parser.add_argument('--output_noise', type=float, default=0.001)
    parser.add_argument('--init_prior_var', type=float, default=1.0)
    parser.add_argument('--vi', action='store_true')
    parser.add_argument('--map_mean', action='store_true')
    parser.add_argument('--measure_dist', type=str, default="uniform")
    parser.add_argument('--measure_scale', type=float, default=1.0)
    parser.add_argument('--sym_steps', type=int, default=20)
    parser.add_argument('--bch_order', type=int, default=1)
    parser.add_argument('--sym_samples', type=int, default=16)
    parser.add_argument('--sym_over_path', action='store_true')
    parser.add_argument('--stl', action='store_true')
    parser.add_argument('--mc_samples', type=int, default=20)
    parser.add_argument('--output_var_method', type=str, default='fixed')
    parser.add_argument('--prior_prec_method', type=str, default='gradients')
    parser.add_argument('--init_posterior_scale', type=float, default=10.0)
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--conserved_init_std', type=float, default=0.0)
    parser.add_argument('--load_model', type=str, default='')

    parser.add_argument('--residual', action='store_true')

    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--multisteps', type=int, default=1)
    parser.add_argument('--q_scale', type=float, default=1.0)
    parser.add_argument('--p_scale', type=float, default=0.5)
    parser.add_argument('--q_shift', type=float, default=0.0)
    parser.add_argument('--p_shift', type=float, default=0.0)

    args = parser.parse_args()
    main(args)

