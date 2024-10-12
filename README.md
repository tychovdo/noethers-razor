# Noether's razor

Code for NeurIPS 2024 paper: "Noether's razor: Learning Conserved Quantities"

Arxiv link: [https://arxiv.org/abs/2410.08087](https://arxiv.org/abs/2410.08087)

### Example usage

```
python main.py --group quadratic_learn1 --train_size 7 --train_steps 4 --steps 20 --stepsize 0.20 --sym_steps 50 --vi --dynamics harmonic-oscillator --sym_samples 200 --sym_over_path --n_epochs 5000
```

### Dynamical systems

New dynamical systems can be added by inheriting the `Dynamics` class in `dynamics.py`, which only requires specifying the Hamiltonian `dynamics.H()` of the system. Optionally, plotting functions can be added to this class. See implementations of existing dynamical systems as reference.

### Use data generator in your project

We tried to keep data generation relatively independent of the other code in the project, so it can be reused to evaluate other learning algorithms.

The code below shows an example of generation 4-body simulation in 3-dimensions:

```
from dynamics import Nbody
from data import generate_dataset

dynamics = Nbody(dim=3, n_bodies=4)

trajectories = generate_dataset(dynamics.initial_phase, n_trajectories=10, n_steps=20, stepsize=0.1)

print(trajectories.shape) # returns shape (10, 20, 2*3*4) with flattened phase space in last dim.
print(trajectories.reshape(10, 20, 2, 3, 4)) # optionally reshaped to (n_trajectories, n_steps, 2, dim, n_bodies) with '2' for momenta/positions.
```

The generation code also supports automatic saving/loading using `save_name` argument.

### Citation

```
@article{van2024noether,
  title={Noether's razor: Learning Conserved Quantities},
  author={van der Ouderaa, Tycho F. A and van der Wilk, Mark and de Haan, Pim},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
