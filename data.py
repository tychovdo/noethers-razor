from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from odeint import ode_int

from utils import symplectic_form


def generate_dataset(hamiltonian, initial_fn, save_name=None, n_trajectories=1000, n_steps=50, stepsize=0.1, seed=100):
    dataset_dir = "data"
    dataset_name = f"data_{save_name}_ntrajectories={n_trajectories}_steps={n_steps}_stepsize={stepsize}_seed={seed}"

    Path(dataset_dir).mkdir(parents=True, exist_ok=True)    

    key = jax.random.PRNGKey(seed)

    try:
        trajectories = jnp.load(f"{dataset_dir}/{dataset_name}.npy")

        print(f"Loaded dataset: {dataset_name}")

        return trajectories
    except:
        print(f"Failed to loader dataset: {dataset_name}")

    print(f"Generating dataset... {dataset_name}")
    trajectories = []

    from tqdm import tqdm
    for i in tqdm(range(n_trajectories)):
        datasteps = n_steps * i

        # Initial conditions
        key, subkey = jax.random.split(key)
        x_start = initial_fn(subkey)

        x_start = x_start.reshape(-1) # M = 2 * n_objects * dim

        jac_h = jax.grad(hamiltonian)
        grad_x = lambda none1,none2,x,none3: symplectic_form(jac_h(x))

        # Simulate
        trajectory, _ = generate_trajectory(grad_x, x_start, stepsize, n_steps=n_steps) # (T, M)

        # Append to trajectory list
        trajectories.append(trajectory)

    trajectories = jnp.stack(trajectories)

    jnp.save(f"{dataset_dir}/{dataset_name}.npy", trajectories)

    return trajectories


def generate_trajectory(grad_x, x_start, stepsize, n_steps=500):
    t_start = 0.0
    t_end = n_steps * stepsize

    t_span = jnp.linspace(t_start, t_end, n_steps + 1)

    solution = ode_int(None, None, grad_x, x_start, t_span, backend='experimental', atol=1e-10, rtol=1e-10) # (T, M)

    return solution, t_span

    
