import io
from functools import partial
from PIL import Image

import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt

def generator_of_quadratic(W, b):
    M = len(b)
    zeros = jnp.zeros((1, M))
    return jnp.block([[symplectic_matrix(W + W.T), symplectic_form(b).reshape(M, 1)], [zeros, 0]])

def plot_matrix(matrix):
    figwidth = len(matrix) / 8 * 5
    fig, ax = plt.subplots(figsize=(figwidth, figwidth))

    ax.imshow(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def lower_triangular(matrix, eps=1+1e-8):
    out = jnp.tril(matrix, -1) + jnp.diag(jnp.abs(jnp.diag(matrix)) * eps)

    return out * 2.0

def _raise_not_finite(x):
  if not jnp.isfinite(x):
    raise ValueError("array is not finite")

def add_nested_dicts(dict1, dict2):
    # Ensure we apply 'add_tensors' at the leaves of the nested structure
    def apply_fn(x, y):
        if isinstance(x, dict):
            # Recurse into the dictionary
            return {k: apply_fn(x[k], y[k]) for k in x}
        else:
            # Apply the leaf operation
            return add_tensors(x, y)
    return apply_fn(dict1, dict2)

def fuse_grads(grads1, grads2):
    if type(grads1) == dict:
        return add_nested_dicts
    else:
        return lambda grads1, grads2: jax.tree_map(lambda u, v: u+v, grads1, grads2)

def symplectic_form(x):
    """ Returns symplectic form of a function

        Input: x (M,)
        Returns: symplectic form (M,)
        """
    assert len(x.shape) == 1, f"symplectic form expects a Jacobian of shape (M,). Got: {x.shape}."
    assert (len(x) % 2) == 0, f"input shape should be even. Got {x.shape}."

    D = x.shape[0] // 2

    q, p = x[:D], x[D:]
    
    return jnp.concatenate([p, -q]) # Hamilton's equation of motion

def symplectic_matrix(x):
    """ Returns symplectic form of a matrix J X

        Input: matrix X (M, M)
        Returns: symplectic form JX of shape (M,M)
        """
    assert len(x.shape) == 2, f"symplectic form expects a Jacobian of shape (M, M). Got: {x.shape}."
    assert (x.shape[0] % 2) == 0, f"input shape should be even. Got {x.shape}."
    assert (x.shape[1] % 2) == 0, f"input shape should be even. Got {x.shape}."

    D = x.shape[0] // 2

    q, p = x[:D, :], x[D:, :]

    return jnp.concatenate([p, -q], axis=0)


def get_hypers(hypers, learn_prior_prec=False, learn_output_var=False):
    *sqrt_prior_precs, output_std = hypers

    prior_precs = []
    for sqrt_prior_prec in sqrt_prior_precs:
        if learn_prior_prec:
            prior_precs.append(sqrt_prior_prec ** 2)
        else:
            prior_precs.append(jax.lax.stop_gradient(sqrt_prior_prec ** 2))

    if learn_output_var:
        output_var = output_std ** 2
    else:
        output_var = jax.lax.stop_gradient(output_std ** 2)

    return *prior_precs, output_var
    
def auto_prior_prec(hypers, params, epoch_log, prior_prec_method, update=0.5):
    n_layers = len(params)

    if prior_prec_method == 'empirical':
        *p_all, _ = get_hypers(hypers, learn_prior_prec=False, learn_output_var=False)

        for layer_i, (layer, old_prior_prec) in enumerate(zip(params, p_all)):
            if 'fixed_mean' in layer:
                #print(f"skipping deterministic layer (layer index: {layer_i}).")
                continue

            M_l = layer['mean']
            S_l = layer['S']
            A_l = layer['A']

            L_s = lower_triangular(S_l)
            L_a = lower_triangular(A_l)

            d_s, d_a = M_l.shape
            d = d_s * d_a

            target_prior_prec = d / (jnp.sum(L_s ** 2) * jnp.sum(L_a ** 2) + jnp.sum(M_l ** 2))

            new_prior_prec = (1 - update) * old_prior_prec + update * target_prior_prec

            hypers[-(n_layers + 1) + layer_i] = jnp.sqrt(jax.lax.stop_gradient(new_prior_prec))

        *p_all, _ = get_hypers(hypers, learn_prior_prec=False, learn_output_var=False)
    elif prior_prec_method in ['fixed', 'gradients']:
        pass
    else:
        raise NotImplementedError(f"Unknown selection for prior variance: {prior_prec_method} ")

def auto_output_var(hypers, params, epoch_log, output_var_method, update=0.02, min_output_var=0.001, max_output_var=0.05):
    if output_var_method == 'empirical':
        *_, old_mse = get_hypers(hypers, learn_prior_prec=False, learn_output_var=False)

        mses = epoch_log["mse"]

        target_mse = sum(mses) / len(mses)

        new_mse = (1 - update) * old_mse + update * target_mse
        new_mse = max(min_output_var, new_mse)
        new_mse = min(max_output_var, new_mse)

        hypers[-1] = jnp.sqrt(new_mse)
    elif output_var_method == 'map_empirical':
        *_, old_mse = get_hypers(hypers, learn_prior_prec=False, learn_output_var=False)

        mses = epoch_log["map_mse"]
        target_mse = sum(mses) / len(mses)

        new_mse = (1 - update) * old_mse + update * target_mse
        new_mse = max(min_output_var, new_mse)
        new_mse = min(max_output_var, new_mse)

        hypers[-1] = jnp.sqrt(new_mse)
    elif output_var_method in ['fixed', 'gradients']:
        pass
    else:   
        raise NotImplementedError(f"Unknown selection for likelihood variance: {output_var_method} ")

