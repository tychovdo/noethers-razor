import numpy as np
from itertools import combinations
from functools import partial

import jax
import jax.numpy as jnp
import diffrax

import math

from utils import symplectic_form, symplectic_matrix, lower_triangular, generator_of_quadratic

from odeint import ode_int

from typing import Callable

SUPPORTED_GROUPS = ['t2', 'so2', 'se2']


def batch_hnn_H(conserved, group, canonicaliser, residual, measure_dist, measure_scale, sym_samples, sym_over_path, n_layers, act_fun, sym_steps, bch_order, tmp, sym_key, weights, cypers, batch_x):
    if canonicaliser is not None:
        batch_x = jax.vmap(canonicaliser)(batch_x)

    f_model = partial(batch_mlp, weights, act_fun, residual)

    if 'quadratic' in group:
        return affine_symmetrised_hnn_H(weights, cypers, conserved, measure_dist, measure_scale, sym_samples, tmp, sym_key, f_model, batch_x, sym_steps=sym_steps, bch_order=bch_order)
    if 'mlp' in group:
        return symmetrised_hnn_H(weights, cypers, conserved, measure_dist, measure_scale, sym_samples, sym_over_path, sym_key, f_model, act_fun, batch_x, sym_steps=sym_steps, bch_order=bch_order)
    elif group == '':
        return f_model(batch_x)
    else:
        raise NotImplementedError(f"Unknown symmetrisation group selected: {group}")

def affine_symmetrised_hnn_H(weights, cypers, conserved, measure_dist, measure_scale, sym_samples, tmp, key, f_model, batch_x, sym_steps=40, bch_order=1):
    """ batch_x : (B, M) -> (,) returns summed Hamiltonian over batch. (for efficiency reasons) """
    assert len(batch_x.shape) == 2, f"Assumes dimension (B, M)... Got {batch_x.shape}"
    assert len(conserved) > 0, f"Requires more than one conserved quantity functions. Got {len(conserved)}."
    assert len(cypers) > 0, f"Requires more than one conserved quantity parameters. Got {len(cypers)}."
    assert len(conserved) == len(cypers), f"Conserved parameters does not match num of conserved functions. Got {len(cypers)}!={len(conserved)}."

    BT, M = batch_x.shape

    group_dim = len(conserved)

    extra_one = jnp.ones((BT, 1))
    batch_x_extra = jnp.concatenate((batch_x, extra_one), 1) # (BT, M+1)

    zeros = jnp.zeros((1, M))

    def sample_weights(key, size):
        if measure_dist == 'uniform':
            return jax.random.uniform(key=key, shape=(size,)) * 2 - 1
        elif measure_dist in ['ball', '2ball']:
            return jax.random.ball(key=key, d=size)
        elif measure_dist == '1ball':
            return jax.random.ball(key=key, d=size, p=1)
        elif measure_dist == 'one':
            return jax.nn.one_hot(jax.random.randint(key=key, shape=(1,), minval=0, maxval=size), size).flatten()
        elif measure_dist in ['gaussian', 'normal']:
            return jax.random.normal(key=key, shape=(size,))

        raise NotImplementedError(f"Unknown group measure: {measure_dist}")

    Gs = jnp.stack([generator_of_quadratic(cyper[0]['W'], cyper[0]['b']) for cyper in cypers])

    sym_keys = jax.random.split(key, num=sym_samples)

    def h_int(sym_key):
        if bch_order == 1:
            weights = sample_weights(sym_key, group_dim).reshape(-1, 1, 1)
            Z = jnp.sum(Gs * weights, 0)
            expZ = jax.scipy.linalg.expm(jnp.pi * Z)
        elif bch_order == 2:
            index_pairs = list(combinations(range(group_dim), 2))

            list_i, list_j = zip(*index_pairs)
            Gs_comm = jax.vmap(lambda i, j: Gs[i] @ Gs[j] - Gs[j] @ Gs[i])(jnp.array(list_i), jnp.array(list_j))

            Gs_comm = Gs_comm.reshape(len(index_pairs), M+1, M+1) # (K(K-1)/2, M+1, M+1)
            Gs_all = jnp.concatenate([Gs, Gs_comm], 0) # (K + K(K-1)/2, M+1, M+1)

            weights = sample_weights(sym_key, len(Gs_all)).reshape(-1, 1, 1) # (K+K(K-1)/2, 1, 1)
            Z = jnp.sum(Gs_all * weights, 0)
            expZ = jax.scipy.linalg.expm(jnp.pi * Z)
        else:
            raise NotImplementedError(f"BCH order of {bch_order} not supported.")


        batch_x_aug = batch_x_extra @ expZ.T
        batch_x_aug = batch_x_aug.reshape(BT, M+1)[:, :-1]

        return f_model(batch_x_aug)

    h_ints = jnp.mean(jax.vmap(h_int)(sym_keys), axis=0) # (BT, 1?)
       
    return h_ints

def symmetrised_hnn_H(weights, cypers, conserved, measure_scale, sym_samples, sym_over_path, key, f_model, act_fun, batch_x, sym_steps=40):
    """ batch_H : (B, M) -> (,) returns summed Hamiltonian over batch. (for efficiency reasons) """
    assert len(batch_x.shape) == 2, f"Assumes dimension (B, M)... Got {batch_x.shape}"
    assert len(conserved) > 0, f"Requires more than one conserved quantity functions. Got {len(conserved)}."
    assert len(cypers) > 0, f"Requires more than one conserved quantity parameters. Got {len(cypers)}."

    BT, M = batch_x.shape

    sym_keys = jax.random.split(key, num=sym_samples)
    signs = jnp.where(jnp.arange(sym_samples) % 2 == 0, 1, -1) # recommended to use even number of sym_samples...
    #signs = jnp.ones(sym_samples)

    h_key, key = jax.random.split(key)

    def h_int(sym_key, sign):
        if sym_over_path:
            sampled_weights = jax.random.normal(key=sym_key, shape=(len(conserved),))
            sampled_weights = sampled_weights / jnp.linalg.norm(sampled_weights)

            C_func = lambda x: jnp.sum(jnp.stack([cons(cyper, x) * weight * measure_scale * sign for cons, cyper, weight in zip(conserved, cypers, sampled_weights)], 0), 0)

            batch_h = jnp.zeros((BT, 1))

            init_x_aug = jnp.concatenate((batch_h,  batch_x), 1) # (BT, 1+M)

            def flat_batch(weights, cypers, flat_batch_x_aug, t):
                x_orig = flat_batch_x_aug.reshape(BT, 1+M)[:, 1:1+M] # (BT, M,)

                jac_C_xorig = jax.vmap(jax.grad(C_func))(x_orig) # (BT, M, 1)
                x_grad = jax.vmap(symplectic_form)(jac_C_xorig) # (BT, M, )

                #measure = 1 / jnp.linalg.norm(sampled_weights)
                #measure = jax.scipy.stats.norm.pdf(t, loc=0, scale=jnp.linalg.norm(sampled_weights))
                measure = jax.scipy.stats.norm.pdf(t, loc=0, scale=1.0)

                h = f_model(x_orig) * measure # (BT, 1)

                return jnp.concatenate([h, x_grad], 1).reshape(BT * (1+M)) # (BT*(1+M),)

            t_span = jnp.array([0.0, 1.0])
            flat_x_init_aug = init_x_aug.reshape(BT*(1+M))

            ode_solution = ode_int(weights, cypers, flat_batch, flat_x_init_aug, t_span, backend='diffrax_direct', adjoint=diffrax.DirectAdjoint(), steps=sym_steps).reshape(BT, 1+M)

            assert ode_solution.shape == (BT, 1+M), f"Wrong shape: expecting solution of shape {(BT, 1+M)}. Got {ode_solution.shape}"

            return ode_solution[:, :1] #/ np.abs(T) # (BT, 1)
        else:
            v = jax.random.uniform(key=sym_key, shape=(len(conserved),)) 

            init_x = batch_x # (BT, M)

            def flat_batch(weights, cypers, flat_batch_x, t): # (BT*M) -> (BT*M)
                measure = 1.0

                x_orig = flat_batch_x.reshape(BT, M) # (BT, M,)

                jac_C_xorig = jax.vmap(jax.grad(C_func))(x_orig) # (BT, M, )
                x_grad = jax.vmap(symplectic_form)(jac_C_xorig) # (BT, M, )

                return x_grad.reshape(BT*M) # (BT*M)

            t_span = jnp.array([0.0, 1.0])
            flat_x_init = init_x.reshape(BT*M)

            ode_solution = ode_int(weights, cypers, flat_batch, flat_x_init, t_span, backend='diffrax_direct', adjoint=diffrax.DirectAdjoint(), steps=sym_steps).reshape(BT, M)
            return f_model(ode_solution) # (BT, 1)

    h_ints = jnp.mean(jax.vmap(h_int)(sym_keys, signs), axis=0) # (BT, 1?)
       
    return h_ints

def summed_model(model, sym_key, weights, cypers, x):
    return model(sym_key, weights, cypers, x).sum()

def predict_grads(weights, cypers, h_model, sym_key, batch_x):
    jac_Hx = jax.grad(partial(summed_model, h_model, sym_key, weights, cypers))(batch_x)

    return jax.vmap(symplectic_form)(jac_Hx)


def sample_weights(params, key, use_mean=False, stop_mean_grad=False, stop_cov_grad=False):
    weights = []

    for layer_i, layer in enumerate(params):
        if 'fixed_mean' in layer:
            M_l = jax.lax.stop_gradient(layer['fixed_mean'])
        elif 'mean' in layer:
            M_l = layer['mean']
        else:
            raise ValueError(f"Layer has no mean? {layer.keys()}")

        if 'fixed_bias' in layer:
            b_l = jax.lax.stop_gradient(layer['fixed_bias'])
        elif 'bias' in layer:
            b_l = layer['bias']
        else:
            b_l = None

        if 'scale' in layer:
            s_l = layer['scale']
        else:
            s_l = None

        if ('S' not in layer) or use_mean:
            W_l = M_l
        else:
            S_l = layer['S']
            A_l = layer['A']
            S_l = jax.lax.stop_gradient(S_l)

            S_l, A_l = lower_triangular(S_l), lower_triangular(A_l)
            if stop_mean_grad:
                M_l = jax.lax.stop_gradient(M_l)

            if stop_cov_grad:
                S_l = jax.lax.stop_gradient(S_l)
                A_l = jax.lax.stop_gradient(A_l)
    
            subkey, key = jax.random.split(key)
            W_l = M_l + S_l @ jax.random.normal(key=subkey, shape=M_l.shape) @ A_l

        sampled_l = {'weight': W_l}

        if s_l is not None:
            sampled_l['scale'] = s_l

        if b_l is not None:
            sampled_l['bias'] = b_l

        weights.append(sampled_l)

    return weights


def batch_mlp(weights, act_fun, residual, batch_x):
    if act_fun == 'elu':
        actfun = jax.nn.elu
    elif act_fun == 'elu2':
        actfun = partial(jax.nn.elu, alpha=2.0)
    elif act_fun == 'elu5':
        actfun = partial(jax.nn.elu, alpha=5.0)
    elif act_fun == 'relu':
        actfun = jax.nn.relu
    elif act_fun == 'cos':
        actfun = jax.lax.cos
    else:
        raise NotImplementedError(f"Unknown activation function: {actfun}")

    z = batch_x
    n_layers = len(weights)

    for layer_i, sampled_layer in enumerate(weights):
        W_l = sampled_layer['weight']
        r = z @ W_l.T

        if 'bias' in sampled_layer:
            b_l = sampled_layer['bias']
            r = r + b_l.reshape(1, -1)

        if 'scale' in sampled_layer:
            scale_l = sampled_layer['scale']
            r = r * scale_l

        if layer_i != (n_layers - 1):
            r = actfun(r)

        if residual and (layer_i != 0) and (layer_i != (n_layers - 1)): # residual only middle layers
            z = z + r
        else:
            z = r

    return z

def init_mlp(key, in_features, n_features, out_features, n_layers, fixed_basis):
    assert n_layers >= 2, f"MLP should have at least 2 layers. Got {n_layers}."

    # layer 0
    gain0 = 1.0
    key_W0, key_b0, key = jax.random.split(key, 3)
    W_0 = jax.random.normal(key_W0, [n_features, in_features]) * jnp.sqrt(1/in_features) * gain0
    b_0 = jax.random.uniform(key_b0, [n_features]) * 2.0 * jnp.pi
    if fixed_basis:
        s_0 = jnp.ones(1) * 0.5
        layers = [{'fixed_mean': W_0, 'fixed_bias': b_0, 'scale': s_0}]
    else:
        layers = [{'mean': W_0, 'bias': b_0}]

    gain = 2.05 # ensure activations at initialisation roughly have a std of 1 across layers
    # L-2 middle layers
    for _ in range(1, n_layers - 1):
        key_Wl, key_bl, key = jax.random.split(key, 3)
        W_l = jax.random.normal(key_Wl, [n_features, n_features]) * gain * jnp.sqrt(1/(n_features + n_features))
        b_l = jax.random.uniform(key_bl, [n_features]) * 2.0 * jnp.pi
        layers += [{'mean': W_l, 'bias': b_l}]

    # layer L
    W_L = jax.random.normal(key, [out_features, n_features]) #* jnp.sqrt(1 / (n_features + out_features))
    layers += [{'mean': W_L}]

    return layers


def init_mlp_vi(key, in_features, n_features, out_features, n_layers, fixed_basis, fixed_basis_scale):
    assert n_layers >= 2, f"MLP should have at least 2 layers. Got {n_layers}."

    # layer 0
    key_M0, key_b0, key = jax.random.split(key, 3)
    M_0 = jax.random.normal(key_M0, [n_features, in_features]) 
    b_0 = jax.random.uniform(key_b0, [n_features]) * 2.0 * jnp.pi * (1.0)

    s_0 = jnp.ones(1) * fixed_basis_scale

    if fixed_basis:
        layers = [{'fixed_mean': M_0, 'fixed_bias': b_0, 'scale': s_0}]
    else:
        std_l = jnp.sqrt(jnp.sqrt(1 / n_features))

        S_0_key, A_0_key, key = jax.random.split(key, 3)
        S_0 = jax.random.normal(S_0_key, [n_features, n_features]) * 0.01 * std_l + jnp.eye(n_features) * std_l
        A_0 = jax.random.normal(A_0_key, [in_features, in_features]) * 0.01 * std_l + jnp.eye(in_features) * std_l

        layers = [{'mean': M_0, 'S': S_0, 'A': A_0, 'bias': b_0, 'scale': s_0}]

    # middle layers
    for _ in range(1, n_layers - 1):
        gain = 1.0

        key_M, key_b, key = jax.random.split(key, 3)
        M_l = jax.random.normal(key_M, [n_features, n_features]) * gain * jnp.sqrt(1/(n_features + n_features))

        std_l = 1.0 * jnp.sqrt(1/(n_features + n_features))

        S_l_key, A_l_key, key = jax.random.split(key, 3)
        S_l = jax.random.normal(S_l_key, [n_features, n_features]) * 0.01 * std_l + jnp.eye(n_features) * std_l
        A_l = jax.random.normal(A_l_key, [n_features, n_features]) * 0.01 * std_l + jnp.eye(n_features) * std_l

        b_l = jax.random.uniform(key_b, [n_features]) * 2.0 * jnp.pi
        layers += [{'mean': M_l, 'S': S_l, 'A': A_l, 'bias': b_l}]

    # layer L
    if fixed_basis:
        M_L = jax.random.normal(key, [out_features, n_features]) * jnp.sqrt(1 / (n_features + out_features)) * 2.0

        std_L = 4.0 * jnp.sqrt(2) * jnp.sqrt(1 / (n_features + out_features))
    else:
        M_L = jax.random.normal(key, [out_features, n_features]) * jnp.sqrt(1 / (n_features + out_features))

        std_L = 4.0 * jnp.sqrt(2) * jnp.sqrt(1 / (n_features + out_features))

    S_L_key, A_L_key, key = jax.random.split(key, 3)
    S_L = jax.random.normal(S_L_key, [out_features, out_features]) * 0.01 * std_L + jnp.eye(out_features) * std_L
    A_L = jax.random.normal(A_L_key, [n_features, n_features]) * 0.01 * std_L + jnp.eye(n_features) * std_L

    if fixed_basis:
        layers += [{'mean': M_L, 'S': S_L, 'A': A_L}]
    else:
        layers += [{'mean': M_L, 'S': S_L, 'A': A_L}]

    return layers


def init_mlp_vi_residual(key, in_features, n_features, out_features, n_layers, fixed_basis):
    assert n_layers >= 2, f"MLP should have at least 2 layers. Got {n_layers}."

    # layer 0
    key_M0, key_b0, key = jax.random.split(key, 3)
    std_l = jnp.sqrt(1 / n_features)
    M_0 = jax.random.normal(key_M0, [n_features, in_features]) * std_l
    b_0 = jax.random.uniform(key_b0, [n_features]) * 2.0 * jnp.pi

    if fixed_basis:
        s_0 = jnp.ones(1) * 0.1
        layers = [{'fixed_mean': M_0, 'fixed_bias': b_0, 'scale': s_0}]
    else:
        S_0_key, A_0_key, key = jax.random.split(key, 3)
        S_0 = jax.random.normal(S_0_key, [n_features, n_features]) * 0.01 * jnp.sqrt(std_l) + jnp.eye(n_features) * jnp.sqrt(std_l)
        A_0 = jax.random.normal(A_0_key, [in_features, in_features]) * 0.01 * jnp.sqrt(std_l) + jnp.eye(in_features) * jnp.sqrt(std_l)

        layers = [{'mean': M_0, 'S': S_0, 'A': A_0, 'bias': b_0}]

    # middle layers
    for _ in range(1, n_layers - 1):
        gain = 1.0

        key_M, key_b, key = jax.random.split(key, 3)
        std_l = jnp.sqrt(1 / (n_features + n_features)) * 0.001 # small because residual
        M_l = jax.random.normal(key_M, [n_features, n_features]) * gain * std_l

        S_l_key, A_l_key, key = jax.random.split(key, 3)
        S_l = jax.random.normal(S_l_key, [n_features, n_features]) * 0.01 * jnp.sqrt(std_l) + jnp.eye(n_features) * jnp.sqrt(std_l)
        A_l = jax.random.normal(A_l_key, [n_features, n_features]) * 0.01 * jnp.sqrt(std_l) + jnp.eye(n_features) * jnp.sqrt(std_l)

        b_l = jax.random.uniform(key_b, [n_features]) * 2.0 * jnp.pi
        layers += [{'mean': M_l, 'S': S_l, 'A': A_l, 'bias': b_l}]

    # layer L
    std_L = jnp.sqrt(2 / (out_features + n_features))
    M_L = jax.random.normal(key, [out_features, n_features]) * std_L

    S_L_key, A_L_key, key = jax.random.split(key, 3)
    S_L = jax.random.normal(S_L_key, [out_features, out_features]) * 0.01 * jnp.sqrt(std_L) + jnp.eye(out_features) * jnp.sqrt(std_L)
    A_L = jax.random.normal(A_L_key, [n_features, n_features]) * 0.01 * jnp.sqrt(std_L) + jnp.eye(n_features) * jnp.sqrt(std_L)

    layers += [{'mean': M_L, 'S': S_L, 'A': A_L}]

    return layers

