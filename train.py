from pathlib import Path
import argparse
import numpy as np
import copy
import psutil
import sys
import gc
import functools
import matplotlib
from collections import defaultdict

from functools import partial

import wandb

import jax
import jax.numpy as jnp
import diffrax
import optax

from jax._src.typing import Array

from jax.scipy.linalg import solve_triangular

from odeint import ode_int 
from utils import symplectic_form, symplectic_matrix, plot_matrix, fuse_grads, get_hypers, lower_triangular, _raise_not_finite
from utils import auto_prior_prec, auto_output_var
from utils import generator_of_quadratic
from network import predict_grads, batch_hnn_H, sample_weights 
from eval import eval_model

from jax import config
# config.update("jax_debug_nans", True)
# config.update("jax_debug_infs", True)
# config.update('jax_disable_jit', True)

import os


def log_normal(x, mu, var, reduce='mean'):
    n, d = x.shape

    log_pdf = -0.5 * n * d * jnp.log(2 * jnp.pi * var) - 0.5 * jnp.sum((x - mu) ** 2) / var
 
    if reduce == 'mean':
        return log_pdf / n
    else:
        return log_pdf

def log_matrix_normal(x, mu, L_s, L_a):
    # UNIT TEST THIS:
    d_s, d_a = x.shape

    # size = jnp.norm(L_s) * jnp.norm(L_a)
    # L_s = L_s / jnp.norm(L_s)
    # L_a = L_a / jnp.norm(L_a)
    
    log_det_L_s = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(L_s))))
    log_det_L_a = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(L_a))))

    Y = x - mu
    W = solve_triangular(L_s, Y, lower=True, trans='T')
    Z = solve_triangular(L_a, W.T, lower=True, trans='T')
    trace_term = jnp.sum(Z ** 2)
    
    log_pdf = -0.5 * d_s * d_a * jnp.log(2 * jnp.pi) - 0.5 * d_a * log_det_L_s - 0.5 * d_s * log_det_L_a - 0.5 * trace_term
    
    return log_pdf

def predict(weights, cypers, key, h_model, batch_x, steps, stepsize):
    t_span = jnp.array([0.0, stepsize])

    BT, M = batch_x.shape

    def flat_batch_predict_grads(weights, cypers, flat_batch_x, t):
        return predict_grads(weights, cypers, h_model, key, flat_batch_x.reshape(BT, M)).reshape(-1)

    flat_batch_x = batch_x.reshape(-1)

    #return ode_int(weights, cypers, flat_batch_predict_grads, flat_batch_x, t_span=t_span, steps=steps, adjoint=diffrax.DirectAdjoint(), backend='diffrax_direct').reshape(BT, M)
    return ode_int(weights, cypers, flat_batch_predict_grads, flat_batch_x, t_span=t_span, steps=steps, adjoint=diffrax.RecursiveCheckpointAdjoint(), backend='diffrax_direct').reshape(BT, M)


def trajectory_to_batches(trajectory, key, noise_std=0.0):
    batch_x = trajectory[:, :-1]
    batch_y = trajectory[:, 1:]

    B, T, M = batch_x.shape

    batch_x = batch_x.reshape(B*T, M)
    batch_y = batch_y.reshape(B*T, M) + jax.random.normal(key, shape=(B*T, M)) * noise_std

    return batch_x, batch_y


@partial(jax.jit, static_argnames=['h_model', 'dataset_size', 'steps'])
def make_map_step(all_params, key, h_model, dataset_size, batch_x, batch_y, steps, kl_amount, stepsize):

    def map_loss(all_params, key, kl_amount):
        params, hypers, cypers = all_params

        n_layers = len(hypers) - 1

        # negative log likelihood (nll)
        weights = sample_weights(params, key=None, use_mean=True)
        pred_y = predict(weights, cypers, key, h_model, batch_x, steps, stepsize)

        n, d = batch_x.shape
        mse = jnp.sum((batch_y - pred_y) ** 2) / (n * d)

        *p_all, output_var = get_hypers(hypers, learn_prior_prec=False, learn_output_var=False)

        nll_loss = -log_normal(pred_y, batch_y, output_var, reduce='mean')

        # negative log prior (nlp)
        nlp_loss = [-log_normal(layer['mean'], 0, 1 / prior_prec) for layer, prior_prec in zip(params, p_all) if not 'fixed_mean' in layer]

        nlp_loss = sum([x.sum() for x in nlp_loss]) / dataset_size

        total_nlp = nlp_loss * kl_amount

        # log
        log_dict = {}
        log_dict['map_joint'] = nll_loss + total_nlp
        log_dict['map_nll'] = nll_loss
        log_dict['map_prior'] = total_nlp
        log_dict['map_mse'] = mse

        return nll_loss + total_nlp, log_dict

    params, hypers, cypers = all_params

    return jax.value_and_grad(map_loss, has_aux=True)(all_params, key, kl_amount)

@partial(jax.jit, static_argnames=['h_model', 'dataset_size', 'steps', 'map_mean', 'learn_output_var', 'learn_prior_prec', 'burnin', 'mc_samples'])
def make_vi_step(all_params, key, h_model, dataset_size, batch_x, batch_y, steps, kl_amount, stepsize, map_mean=False, mc_samples=10, learn_output_var=True, learn_prior_prec=True, burnin=False):
    burnin=False

    def elbo(all_params, key, kl_amount):
        params, hypers, cypers = all_params

        n_layers = len(hypers) - 1
        
        *p_all, output_var = partial(get_hypers, learn_output_var=learn_output_var, learn_prior_prec=learn_prior_prec)(hypers)
        n, d = batch_x.shape

        mc_keys = jax.random.split(key, num=mc_samples)

        def mc_mse_nll(key):
            w_key, key = jax.random.split(key)
            weights = sample_weights(params, w_key, use_mean=False, stop_mean_grad=map_mean, stop_cov_grad=burnin)

            sym_key, key = jax.random.split(key)
            pred_y = predict(weights, cypers, sym_key, h_model, batch_x, steps, stepsize)

            mc_mse = jnp.sum((batch_y - pred_y) ** 2) / (n * d)

            mc_nll_loss = -log_normal(pred_y, batch_y, output_var, reduce='mean')

            return mc_mse, mc_nll_loss

        mc_mse, mc_nll_loss = jax.vmap(mc_mse_nll)(mc_keys)
        jax.debug.print("{x}", x=mc_mse)
        mse = jnp.mean(mc_mse)
        nll_loss = jnp.mean(mc_nll_loss)

        def kron_kl(M, L_s, L_a, prior_prec):
            """ Compute KL(q, p) between a matrixvariate distribution MN(M, S, A) = N(vec(M), A kron S) and
                a matrixvariate prior MN(0, I/s_prec, I/a_prec), where s_prec and a_prec denote prior precisions. """
            d_s, d_a = M.shape
            d = d_s * d_a

            L_s = lower_triangular(L_s)
            L_a = lower_triangular(L_a)

            if burnin:
                L_s = jax.lax.stop_gradient(L_s)
                L_a = jax.lax.stop_gradient(L_s)

            L_s = jax.lax.stop_gradient(L_s)

            L_s_diag = jnp.diag(L_s)
            L_a_diag = jnp.diag(L_a)

            q_logdet = d_a * jnp.sum(2 * jnp.log(jnp.abs(L_s_diag))) + d_s * jnp.sum(2 * jnp.log(jnp.abs(L_a_diag)))

            p_logdet = -d * jnp.log(prior_prec)

            term1 = 0.5 * (p_logdet - q_logdet) # t_c
            term2 = 0.5 * -d
            term3 = 0.5 * jnp.sum(L_s ** 2) * jnp.sum(L_a ** 2) * prior_prec # ???
            if map_mean:
                term4 = 0.5 * jnp.sum(jax.lax.stop_gradient(M) ** 2) * prior_prec
            else:
                term4 = 0.5 * jnp.sum(M ** 2) * prior_prec

            # closed form fixed
            tr_q = jnp.sum(L_s ** 2) * jnp.sum(L_a ** 2)
            mm = jnp.sum(M ** 2)

            closed_q_logdet = d_a * jnp.sum(2 * jnp.log(jnp.abs(L_s_diag))) + d_s * jnp.sum(2 * jnp.log(jnp.abs(L_a_diag)))
            closed_p_logdet = d * jnp.log(tr_q + mm)

            closed_term1 = 0.5 * (closed_p_logdet - closed_q_logdet)
            closed_term2 = 0.5 * -d
            closed_term3 = 0.5 * ((d * tr_q) / (tr_q + mm))
            closed_term4 = 0.5 * ((d * mm) / (tr_q + mm))

            terms = [term1, term2, term3, term4, q_logdet, p_logdet, closed_term1, closed_term2, closed_term3, closed_term4, closed_q_logdet, closed_p_logdet]

            return term1 + term2 + term3 + term4, terms

        kls, kl_terms = [], []
        for layer, prior_prec in zip(params, p_all):
            if not 'fixed_mean' in layer: # skipping deterministic layers for KL
                M_l = layer['mean']
                S_l = layer['S']
                A_l = layer['A']

                kl, kl_term = kron_kl(M_l, S_l, A_l, prior_prec)
                kls.append(kl)
                kl_terms.append(kl_term)
            else:
                kls.append(0.0)
                kl_terms.append(None)

        # use total elbo not the per datapoint convention
        total_kl = sum(kls) / dataset_size

        log_dict = {}
        log_dict['nll'] = nll_loss
        log_dict['elbo'] = nll_loss + total_kl
        log_dict['kl'] = total_kl

        for layer_i, kl in enumerate(kls):
            if kl is not None:
                log_dict[f'layer_kl/kl_{layer_i}'] = kl / dataset_size

        for l, terms in enumerate(kl_terms):
            if terms is not None:
                log_dict[f'kl_terms/{l}_term1(vol)'] = terms[0]
                log_dict[f'kl_terms/{l}_term1a(qlogdet)'] = terms[4]
                log_dict[f'kl_terms/{l}_term1b(plogdet)'] = terms[5]
                log_dict[f'kl_terms/{l}_term2(const)'] = terms[1]
                log_dict[f'kl_terms/{l}_term3(cov)'] = terms[2]
                log_dict[f'kl_terms/{l}_term4(mean)'] = terms[3]

        log_dict['mse'] = mse
        log_dict['stats/dataset_size'] = dataset_size

        total_loss = nll_loss + total_kl

        return total_loss, log_dict

    return jax.value_and_grad(elbo, has_aux=True)(all_params, key, kl_amount)



def train_model(params, hypers, cypers, canonicalise, train_loader, test_loaders, train_loader_noshuffle,
                dynamics, group, n_epochs, n_layers, act_fun, 
                steps, stepsize,
                scheduler='cosine', lr_param=0.0001, lr_hyper=0.001, lr_cyper=0.001, n_burnin=0, cyper_burnin=0, vi=False,
                measure_dist='uniform', measure_scale=1.0, sym_samples=20, sym_over_path=False, map_mean=False,
                local_reparam=False, stl=False, output_noise=0.0, output_var_method='fixed', prior_prec_method='gradients',
                sym_steps=20, bch_order=1, mc_samples=10, tmp=False, beta1=0.9, beta2=0.999, residual=False, curriculum=False,
                multisteps=1):
    # Optimizers
    if scheduler == 'cosine':
        def scheduler(lr, step_number, offset=0):
            decay_steps = (n_epochs - offset) * len(train_loader)

            lr_eff = lr * ((jnp.cos(jnp.pi * step_number / decay_steps) + 1) * 0.5)

            return lr_eff
    else:
        scheduler = optax.constant_schedule

    optimizer_param = optax.adam(partial(scheduler, lr_param), b1=beta1, b2=beta2)
    optimizer_hyper = optax.adam(partial(scheduler, lr_hyper), b1=0.9, b2=0.999)
    #optimizer_cyper = optax.adam(partial(scheduler, lr_cyper, offset=cyper_burnin), b1=beta1, b2=beta2)
    optimizer_cyper = optax.adam(partial(scheduler, lr_cyper, offset=cyper_burnin), b1=beta1, b2=beta2)

    if multisteps > 1:
        optimizer_param = optax.MultiSteps(optimizer_param, every_k_schedule=multisteps)
        optimizer_hyper = optax.MultiSteps(optimizer_hyper, every_k_schedule=multisteps)
        optimizer_cyper = optax.MultiSteps(optimizer_cyper, every_k_schedule=multisteps)

    #optimizer_cyper = optax.sgd(partial(scheduler, lr_cyper), momentum=beta1)

    opt_state_param = optimizer_param.init(params)
    opt_state_hyper = optimizer_hyper.init(hypers)
    opt_state_cyper = optimizer_cyper.init(cypers)

    if canonicalise:
        canonicaliser = dynamics.canonicaliser(group)
    else:
        canonicaliser = None

    # Compute dataset size (n_trajectories*n_steps), as losses and eval metrics are normalised per sample
    assert len(train_loader.dataset[0]) == 1, f"Data format check failed: dataloader should return tuple with trajectory tensor"
    sample_shape = train_loader.dataset[0][0].shape
    assert len(sample_shape) == 2, f"Expecting matrix of trajectory samples (time dim, phase dim). Got shape: {sample_shape}."
    assert sample_shape[1] == dynamics.pdim, f"Data phase dim ({batch_shape[1]}) does not match phase dim of dynamics ({dynamics.pdim})"
    n_trajectories = len(train_loader.dataset)
    n_steps = sample_shape[0] - 1
    dataset_size = n_trajectories * n_steps
    print(f"dataset_size: {dataset_size} (n_trajectories: {n_trajectories}, n_steps: {n_steps})")

    # Model
    conserved = dynamics.conserved_functions(group)

    f_model = partial(batch_hnn_H, conserved, group, canonicaliser, residual, measure_dist, 0.0*measure_scale, sym_samples, sym_over_path, n_layers, act_fun, sym_steps, bch_order, tmp)
    h_model = partial(batch_hnn_H, conserved, group, canonicaliser, residual, measure_dist, measure_scale, sym_samples, sym_over_path, n_layers, act_fun, sym_steps, bch_order, tmp)

    key = jax.random.PRNGKey(100)

    traj_key, key = jax.random.split(key)

    # ALSO EVAL BEFORE
    if True:
        epoch_i = 0

        plot = False
        plot_H = False
        evals, plots = eval_model(params, cypers, hypers, train_loader_noshuffle, dynamics, f_model, h_model, dataset_size, steps, stepsize, plot=plot, plot_H=plot_H, train_loader=train_loader_noshuffle)

        for name in evals:
            stats = evals[name]
            mean = jnp.mean(stats)
            sterr = jnp.std(stats) / jnp.sqrt(len(stats)) if jnp.sqrt(len(stats)) > 0 else 0.0

            print(f"\t\ttrain_mean/map_{name}: {mean:.5f} ({sterr:.7f})")

            wandb.log({f"train_mean/mean_{name}": mean}, epoch_i)
            wandb.log({f"train_mean/sterr_{name}": sterr}, epoch_i)

        for name in plots:
            image = wandb.Image(plots[name])
            wandb.log({f"train_{name}": image}, epoch_i)
            matplotlib.pyplot.close()

        for test_loader_name, test_loader in test_loaders.items():
            plot = True
            plot_H = test_loader_name == 'test'
            evals, plots = eval_model(params, cypers, hypers, test_loader, dynamics, f_model, h_model, dataset_size, steps, stepsize, train_loader=train_loader_noshuffle, plot=True, plot_H=plot_H)

            for name in evals:
                stats = evals[name]
                mean = jnp.mean(stats)
                sterr = jnp.std(stats) / jnp.sqrt(len(stats)) if jnp.sqrt(len(stats)) > 0 else 0.0

                print(f"\t\t{test_loader_name}_mean/map_{name}: {mean:.5f} ({sterr:.7f})")

                wandb.log({f"{test_loader_name}_mean/mean_{name}": mean}, epoch_i)
                wandb.log({f"{test_loader_name}_mean/sterr_{name}": sterr}, epoch_i)

            for name in plots:
                image = wandb.Image(plots[name])
                wandb.log({f"{test_loader_name}_mean_{name}": image}, epoch_i)
                matplotlib.pyplot.close()


    for epoch_i in range(1, n_epochs+1):
        epoch_log = defaultdict(list)

        if epoch_i < n_burnin:
            kl_amount = jnp.cos(jnp.pi * (epoch_i) / n_burnin + jnp.pi) / 2 + 0.5 # cosine-start KL on first 100 epochs (first 10%)
        else:
            kl_amount = 1.0

        for batch_idx, trajectory in enumerate(train_loader):
            *p_all, output_var = get_hypers(hypers)
            batch_x, batch_y = trajectory_to_batches(trajectory, traj_key, noise_std=jnp.sqrt(output_noise)) # sqrt(0.001) = 0.00001

            if vi:
                learn_prior_prec = prior_prec_method == 'gradients'
                learn_output_var = output_var_method == 'gradients'

                if map_mean:
                    subkey, key = jax.random.split(key)

                    (_, log_dict), (param_grads_map, hyper_grads_map, cyper_grads_map) = make_map_step((params, hypers, cypers), subkey, h_model, dataset_size, batch_x, batch_y, steps, kl_amount, stepsize)

                    for log_key, val in log_dict.items():
                        epoch_log[log_key].append(val)

                    key, subkey = jax.random.split(key)

                    (_, log_dict), (param_grads_vi, hyper_grads_vi, cyper_grads_vi) = make_vi_step((params, hypers, cypers), subkey, h_model, dataset_size, batch_x, batch_y, steps, kl_amount, stepsize, map_mean=map_mean, learn_prior_prec=learn_prior_prec, learn_output_var=learn_output_var, burnin=epoch_i < n_burnin, mc_samples=mc_samples)

                    for log_key, val in log_dict.items():
                        epoch_log[log_key].append(val)

                    hyper_grads = hyper_grads_vi
                    param_grads = [dict([(key, layer1[key] + layer2[key]) for key in layer1.keys()]) for layer1, layer2 in zip(param_grads_map, param_grads_vi)]
                    cyper_grads = []

                    for cyper_i in range(len(cyper_grads_vi)):
                        if cyper_grads_vi[cyper_i] is not None:
                            cyper_grads.append([dict([(key, layer1[key] + layer2[key]) for key in layer1.keys()]) for layer1, layer2 in zip(cyper_grads_map[cyper_i], cyper_grads_vi[cyper_i])])
                        else:
                            cyper_grads.append(None)

                else:
                    key, subkey = jax.random.split(key)

                    (_, log_dict), (param_grads, hyper_grads, cyper_grads) = make_vi_step((params, hypers, cypers), subkey, h_model, dataset_size, batch_x, batch_y, steps, kl_amount, stepsize, map_mean=map_mean, learn_prior_prec=learn_prior_prec, learn_output_var=learn_output_var, burnin=epoch_i < n_burnin, mc_samples=mc_samples)

                    for log_key, val in log_dict.items():
                        epoch_log[log_key].append(val)
            else:
                key, subkey = jax.random.split(key)
                (_, epoch_log), (param_grads, hyper_grads, cyper_grads) = make_map_step((params, hypers, cypers), subkey, h_model, dataset_size, batch_x, batch_y, steps, kl_amount, stepsize)

                for log_key, val in log_dict.items():
                    epoch_log[log_key].append(val)

            from jax import tree_util

            def check_tree_finiteness(param_tree):
                is_finite_tree = tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), param_tree)
                all_finite = tree_util.tree_reduce(lambda x, y: x & y, is_finite_tree, True)
                return all_finite

            updates_param, opt_state_param = optimizer_param.update(param_grads, opt_state_param, params)
            params = optax.apply_updates(params, updates_param)

            if epoch_i > cyper_burnin:
                updates_cyper, opt_state_cyper = optimizer_cyper.update(cyper_grads, opt_state_cyper, cypers)
                cypers = optax.apply_updates(cypers, updates_cyper)

            # Automatically set important hyperparameters (derived using EM)
            auto_output_var(hypers, params, epoch_log, output_var_method)
            auto_prior_prec(hypers, params, epoch_log, prior_prec_method)

        # EVAL
        *p_all, output_var = get_hypers(hypers)

        log_str = f'\t[{epoch_i}]'
        for log_name, item in epoch_log.items():
            log_mean = np.mean(item)
            wandb.log({f"train/{log_name}": log_mean}, epoch_i)
            log_str += f"   {log_name}:{log_mean:.3f}"

        #wandb.log({'train/lr_param': float(scheduler(lr_param, opt_state_param[-1][0]))}, epoch_i)
        #wandb.log({'train/lr_hyper': float(scheduler(lr_hyper, opt_state_hyper[-1][0]))}, epoch_i)
        #wandb.log({'train/lr_cyper': float(scheduler(lr_cyper, opt_state_cyper[-1][0]))}, epoch_i)
        wandb.log({'train/kl_amount': kl_amount}, epoch_i)

        for layer_i, (layer, old_prior_prec) in enumerate(zip(params, p_all)):
            if 'fixed_mean' in layer:
                M_l = layer['fixed_mean']
                if 'fixed_bias' in layer:
                    b_l = layer['fixed_bias']
                else:
                    b_l = None
                wandb.log({f"param_norms/scale{layer_i}": layer['scale']}, epoch_i)
            else:
                M_l = layer['mean']
                if 'bias' in layer:
                    b_l = layer['bias']
                else:
                    b_l = None

            wandb.log({f"sum_of_squared/M{layer_i}": jnp.sum(M_l ** 2)}, epoch_i)
            wandb.log({f"param_norms/M{layer_i}": jnp.linalg.norm(M_l)}, epoch_i)

            if b_l is not None:
                wandb.log({f"sum_of_squared/b{layer_i}": jnp.sum(b_l ** 2)}, epoch_i)
                wandb.log({f"param_norms/b{layer_i}": jnp.linalg.norm(b_l)}, epoch_i)

            if 'S' in layer:
                S_l = layer['S']
                A_l = layer['A']

                S_l = lower_triangular(S_l)
                A_l = lower_triangular(A_l)

                wandb.log({f"sum_of_squared/S{layer_i}": jnp.sum(S_l ** 2)}, epoch_i)
                wandb.log({f"sum_of_squared/A{layer_i}": jnp.sum(A_l ** 2)}, epoch_i)
                wandb.log({f"sum_of_squared/SxA{layer_i}": jnp.sum(S_l ** 2) * jnp.sum(A_l ** 2)}, epoch_i)

                wandb.log({f"param_norms/S{layer_i}": jnp.linalg.norm(S_l)}, epoch_i)
                wandb.log({f"param_norms/A{layer_i}": jnp.linalg.norm(A_l)}, epoch_i)

                d_s, d_a = M_l.shape
                d = d_s * d_a

                emp_prior_prec = d / (jnp.sum(S_l ** 2) * jnp.sum(A_l ** 2) + jnp.sum(M_l ** 2))

                wandb.log({f"auto/emp_prior_prec{layer_i}": emp_prior_prec}, epoch_i)

        for ci, cyper in enumerate(cypers):
            if cyper is not None:
                for layer_i, layer in enumerate(cyper):
                    if 'mean' in layer.keys():
                        c_M = layer['mean']
                        wandb.log({f"param_norms/c{ci}_M{layer_i}": jnp.linalg.norm(c_M)}, epoch_i)
                        if 'bias' in layer:
                            c_b = layer['bias']
                            wandb.log({f"param_norms/c{ci}_b{layer_i}": jnp.linalg.norm(c_b)}, epoch_i)

                    if 'W' in layer.keys():
                        if tmp:
                            cypers[ci][0]['W'] = 0.5 * (cypers[ci][0]['W'] + cypers[ci][0]['W'].T)

                        c_M = layer['W']
                        wandb.log({f"param_norms/c{ci}_W{layer_i}": jnp.linalg.norm(c_M)}, epoch_i)
                        if 'b' in layer:
                            c_b = layer['b']
                            wandb.log({f"param_norms/c{ci}_b{layer_i}": jnp.linalg.norm(c_b)}, epoch_i)

                        for i in range(c_M.shape[0]):
                            for j in range(c_M.shape[1]):
                                wandb.log({f"cons_W/c{ci}_W_{i}_{j}": c_M[i, j]}, epoch_i)

                            wandb.log({f"cons_b/c{ci}_b_{i}_{j}": c_b[i]}, epoch_i)

                        # plot generator
                        M = len(c_b)
                        G = generator_of_quadratic(c_M, c_b) * measure_scale

                        plot = plot_matrix(G)
                        image = wandb.Image(plot)
                        wandb.log({f"matrix_G/G_{ci}": image}, epoch_i)
                        matplotlib.pyplot.close()

                        plot = plot_matrix(c_M)
                        image = wandb.Image(plot)
                        wandb.log({f"matrix_W/W_{ci}": image}, epoch_i)
                        matplotlib.pyplot.close()

                        plot = plot_matrix(c_M + c_M.T)
                        image = wandb.Image(plot)
                        wandb.log({f"matrix_SW/W_{ci}": image}, epoch_i)
                        matplotlib.pyplot.close()

                        plot = plot_matrix(symplectic_matrix(c_M + c_M.T))
                        image = wandb.Image(plot)
                        wandb.log({f"matrix_symSW/W_{ci}": image}, epoch_i)
                        matplotlib.pyplot.close()



        for l, p_l in enumerate(p_all):
            wandb.log({f"hyper/prior_prec{l}": p_l}, epoch_i)
        wandb.log({f"hyper/output_var": output_var}, epoch_i)

        print(f"epoch: {epoch_i} / {n_epochs}")

        if (epoch_i % 10) == 0:
            plot = ((epoch_i % 100) == 0)

            evals, plots = eval_model(params, cypers, hypers, train_loader_noshuffle, dynamics, f_model, h_model, dataset_size, steps, stepsize, plot=plot, plot_H=True, train_loader=train_loader_noshuffle)

            for name in evals:
                stats = evals[name]
                mean = jnp.mean(stats)
                sterr = jnp.std(stats) / jnp.sqrt(len(stats)) if jnp.sqrt(len(stats)) > 0 else 0.0

                wandb.log({f"train_mean/mean_{name}": mean}, epoch_i)
                wandb.log({f"train_mean/sterr_{name}": sterr}, epoch_i)

            for name in plots:
                image = wandb.Image(plots[name])
                wandb.log({f"train_{name}": image}, epoch_i)
                matplotlib.pyplot.close()

        for test_loader_name, test_loader in test_loaders.items():
            if (epoch_i % 10) == 0:
                plot = ((epoch_i % 100) == 0)
                plot_H = test_loader_name == 'test'
                evals, plots = eval_model(params, cypers, hypers, test_loader, dynamics, f_model, h_model, dataset_size, steps, stepsize, plot=plot, plot_H=plot_H, train_loader=train_loader_noshuffle)

                for name in evals:
                    stats = evals[name]
                    mean = jnp.mean(stats)
                    sterr = jnp.std(stats) / jnp.sqrt(len(stats)) if jnp.sqrt(len(stats)) > 0 else 0.0

                    print(f"\t\t{test_loader_name}_mean/map_{name}: {mean:.5f} ({sterr:.7f})")

                    wandb.log({f"{test_loader_name}_mean/mean_{name}": mean}, epoch_i)
                    wandb.log({f"{test_loader_name}_mean/sterr_{name}": sterr}, epoch_i)

                for name in plots:
                    image = wandb.Image(plots[name])
                    wandb.log({f"{test_loader_name}_mean_{name}": image}, epoch_i)
                    matplotlib.pyplot.close()

        gc.collect()

    return params, hypers, cypers



