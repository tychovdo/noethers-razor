from collections import defaultdict
import io
import math
from PIL import Image

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from functools import partial

import jax
import jax.numpy as jnp

from odeint import ode_int 
from network import predict_grads, sample_weights
from utils import symplectic_form, get_hypers


def chunked(fn, x, n_chunks=5):
    chunk_size = len(x) // n_chunks # B'

    results = []
    for chunk in [x[i:i + chunk_size] for i in range(0, len(x), chunk_size)]:
        result = fn(chunk)

        results.append(result)

    return jnp.concatenate(results) # (B, M)

@partial(jax.jit, static_argnames=['h_model'])
def eval_phase(params, cypers, output_var, h_model, traj_x, traj_y, t_span):
    key = jax.random.PRNGKey(100) # always same at eval

    weights = sample_weights(params, None, use_mean=True)

    sym_key, key = jax.random.split(key)
    predict_single = lambda weights, cypers, x, t: predict_grads(weights, cypers, h_model, sym_key, x.reshape(1, -1)).reshape(-1)
    
    pred_y = chunked(jax.vmap(partial(ode_int, weights, cypers, predict_single, t_span=t_span, backend='diffrax_direct')), traj_x) # (B, M)

    assert pred_y.shape[1] == 1, f"expect only one output dim"
    pred_y = pred_y[:, 0]

    _, d = traj_x.shape

    mse = jnp.mean((traj_y - pred_y) ** 2, axis=1)

    nll = 0.5 * d * jnp.log(2 * jnp.pi * output_var) + 0.5 * jnp.sum((traj_y - pred_y) ** 2, axis=1) / output_var

    return mse, nll

@partial(jax.jit, static_argnames=['model', 'use_mean_predictor'])
def model_predict(model, w_key, params, cypers, phase_grid, use_mean_predictor=True):
    if use_mean_predictor:
        weights = sample_weights(params, None, use_mean=True)
    else:
        weights = sample_weights(params, w_key)

    sym_key = jax.random.PRNGKey(100) # same symmetrisation at eval
    return model(sym_key, weights, cypers, phase_grid)
    
@partial(jax.jit, static_argnames=['h_model', 'steps'])
def predict_traj(params, cypers, h_model, trajectory, steps, t_span):
    key = jax.random.PRNGKey(100) # always same at eval

    weights = sample_weights(params, None, use_mean=True)

    B, T, M = trajectory.shape

    start_x = trajectory.reshape(B, T, M)[:, 0, :] # (B, M)

    flat_start_x = start_x.reshape(B*M)

    sym_key, key = jax.random.split(key)
    flat_batch_predict = lambda weights, cypers, x, t: predict_grads(weights, cypers, h_model, sym_key, x.reshape(-1, M)).reshape(-1) # x: (B'*M) -> (B'*M)

    #pred_y = ode_int(weights, cypers, flat_batch_predict, flat_start_x, t_span=t_span, backend='experimental')
    pred_y = ode_int(weights, cypers, flat_batch_predict, flat_start_x, t_span=t_span, backend='diffrax_direct_tspan')

    assert pred_y.shape == (len(t_span), B*M), f"Expecting shape (len(t_span), B*M) {(len(t_span), B*M)}. Got {pred_y.shape}."

    pred_y = pred_y.reshape(len(t_span), B, M).transpose(1, 0, 2) # (B, T?, M)

    return pred_y

def plot_traj(trajectory, pred_y, dynamics, t_span):
    B, T, M = pred_y.shape

    # Make Image plot
    W = math.ceil(math.sqrt(B))
    H = math.ceil(B / W)

    fig, axs = plt.subplots(ncols=W, nrows=H, figsize=(W*2, H*2))

    for h in range(H):
        for w in range(W):
            if H == 1: # it's so frustrating that matplotlib makes axs a list if there is only one row...
                ax = axs[w]
            else:
                ax = axs[h, w]
            ax_i = h*W + w

            if ax_i < len(trajectory):
                dynamics.plot_trajectory(trajectory[ax_i], t_span, ax, transparent=True)
                dynamics.plot_trajectory(pred_y[ax_i], t_span, ax, transparent=False)
            else:
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def plot_phase(grid_energy, dynamics, lim=1.0, trajectories=None):
    # Make Image plot
    fig, ax = plt.subplots(figsize=(2, 2))

    dynamics.plot_phase_energy(grid_energy, ax, lim=lim)

    if trajectories is not None:
        dynamics.plot_phase_trajectories(trajectories, ax, lim=lim)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def plot_phase_line(grid_energy, dynamics, lim=1.0, trajectories=None):
    # Make Image plot
    fig, ax = plt.subplots(figsize=(2, 2))

    dynamics.plot_phase_energy_line(grid_energy, ax, lim=lim)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)


@partial(jax.jit, static_argnames=['dynamics', 'h_model'])
def eval_grad(params, cypers, dynamics, h_model, traj_x):
    weights = sample_weights(params, None, use_mean=True)

    sym_key = jax.random.PRNGKey(100) # always same symmetrisation at eval

    pred_grads = chunked(partial(predict_grads, weights, cypers, h_model, sym_key), traj_x)

    traj_jacs = jax.vmap(jax.grad(dynamics.H))(traj_x)

    traj_grads = jax.vmap(symplectic_form)(traj_jacs)

    return jnp.power(pred_grads - traj_grads, 2)

@partial(jax.jit, static_argnames=['h_model', 'B', 'T'])
def eval_H(params, cypers, h_model, traj_x, B, T):
    weights = sample_weights(params, None, use_mean=True)

    key = jax.random.PRNGKey(100) # always same symmetrisation at eval

    pred_H = h_model(key, weights, cypers, traj_x).reshape(B, T)

    pred_H_norm = pred_H - pred_H[:, :1]

    return jnp.power(pred_H_norm, 2)


def eval_model(params, cypers, hypers, data_loader, dynamics, f_model, h_model, dataset_size, steps, stepsize, plot=False, plot_H=False, train_loader=None):
    evals = defaultdict(list)
    plots = defaultdict(list)

    *p_all, output_var = get_hypers(hypers)

    key = jax.random.PRNGKey(100) # always same at eval

    for batch_idx, trajectory in enumerate(data_loader):
        B, T, M = trajectory.shape

        # plot traj
        if plot:
            try:
                if batch_idx == 0:
                    t_span = jnp.linspace(0, stepsize*(T-1), T) # (B, T, M)
                    pred_y = predict_traj(params, cypers, h_model, trajectory, steps, t_span)

                    plots['traj_plot/traj'] = plot_traj(trajectory, pred_y, dynamics, t_span)
            except NotImplementedError:
                print(f"Skipping trajectory plotting as not available for chosen dynamics:")

        # plot H
        if plot_H:
            if hasattr(dynamics, 'plot_phase_energy'):
                if batch_idx == 0:
                    for lim in [3, 8]:
                        H, W = 20, 20
                        y = jnp.linspace(-lim, lim, H)
                        x = jnp.linspace(-lim, lim, W)
                        xx, yy = jnp.meshgrid(x, y)
                        phase_grid = jnp.stack([xx, yy], 2).reshape(H*W, 2) # (HW, 2)

                        true_grid_H = jax.vmap(dynamics.H)(phase_grid)

                        plots[f'H_plot/true_H_lim{lim}'] = plot_phase(true_grid_H.reshape(H, W), dynamics, lim=lim)
                        plots[f'H_line_plot/true_H_lim{lim}'] = plot_phase_line(true_grid_H.reshape(H, W), dynamics, lim=lim)

                        if False:
                            for i in range(2):
                                key_i, key = jax.random.split(key)
                                pred_grid_Fi = model_predict(f_model, key_i, params, cypers, phase_grid, use_mean_predictor=False)
                                pred_grid_Hi = model_predict(h_model, key_i, params, cypers, phase_grid, use_mean_predictor=False)

                                if train_loader is not None:
                                    for train_trajectory in train_loader:
                                        plots[f'F_plot/pred_F_lim{lim}_{i}'] = plot_phase(pred_grid_Fi.reshape(H, W), dynamics, lim=lim, trajectories=train_trajectory)
                                        #plots[f'pred_H_lim{lim}_{i}'] = plot_phase(pred_grid_Hi.reshape(H, W), dynamics, lim=lim, trajectories=train_trajectory)
                                else:
                                    plots[f'F_plot/pred_F_lim{lim}_{i}'] = plot_phase(pred_grid_Fi.reshape(H, W), dynamics, lim=lim)
                                    #plots[f'pred_H_lim{lim}_{i}'] = plot_phase(pred_grid_Hi.reshape(H, W), dynamics, lim=lim)

                        pred_grid_F = model_predict(f_model, key, params, cypers, phase_grid, use_mean_predictor=True)
                        pred_grid_H = model_predict(h_model, key, params, cypers, phase_grid, use_mean_predictor=True)

                        if train_loader is not None:
                            for train_trajectory in train_loader:
                                #plots[f'pred_F_lim{lim}'] = plot_phase(pred_grid_F.reshape(H, W), dynamics, lim=lim, trajectories=train_trajectory)
                                plots[f'H_plot/pred_H_lim{lim}'] = plot_phase(pred_grid_H.reshape(H, W), dynamics, lim=lim, trajectories=train_trajectory)
                                plots[f'H_line_plot/pred_H_lim{lim}'] = plot_phase_line(pred_grid_H.reshape(H, W), dynamics, lim=lim, trajectories=train_trajectory)
                        else:
                            #plots[f'pred_F_lim{lim}'] = plot_phase(pred_grid_F.reshape(H, W), dynamics, lim=lim)
                            plots[f'H_plot/pred_H_lim{lim}'] = plot_phase(pred_grid_H.reshape(H, W), dynamics, lim=lim, trajectories=train_trajectory)
                            plots[f'H_line_plot/pred_H_line_lim{lim}'] = plot_phase_line(pred_grid_H.reshape(H, W), dynamics, lim=lim, trajectories=train_trajectory)

            # else:
            #     print(f"Skipping H plot because phase space dimension is >2 ({dynamics.pdim})")

        t_span = jnp.array([0.0, stepsize])

        traj_x = trajectory[:, :-1]
        traj_y = trajectory[:, 1:]

        traj_x = traj_x.reshape(B*(T-1), M)
        traj_y = traj_y.reshape(B*(T-1), M)

        # eval phase
        phase_mse, phase_nll = eval_phase(params, cypers, output_var, h_model, traj_x, traj_y, t_span)
        evals['mse'].append(phase_mse)
        evals['nll'].append(phase_nll)

        # eval grad
        grad_err = eval_grad(params, cypers, dynamics, h_model, traj_x)
        evals['grad'].append(grad_err)

        # eval H
        key = jax.random.PRNGKey(100) # always same at eval

        traj_H_tmp = jax.vmap(dynamics.H)(traj_x)

        traj_H = jax.vmap(dynamics.H)(traj_x).reshape(B, T-1)
        pred_H = model_predict(h_model, key, params, cypers, traj_x, use_mean_predictor=True)

        H_err = eval_H(params, cypers, h_model, traj_x, B, T-1)
        evals['H'].append(H_err)

    for name in evals.keys():
        evals[name] = np.concatenate(evals[name])

    return evals, plots


