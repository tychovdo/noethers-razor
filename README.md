# Noether's razor

Code for NeurIPS 2024 paper: "Noether's razor: learning conserved quantities" 
by Tycho F. A. van der Ouderaa, Mark van der Wilk and Pim de Haan

### Examples

| Method | Symmetry | Dynamics | Code  | 
|---|---|---|---|
| NN | - | 2d 3-body | `python main.py --dynamics nbody_2d_3b` |
| NN | Oracle SE(2) symmetry | 2d 3-body | `python main.py --dynamics nbody_2d_3b --group se2` |
| NN | Learned symmetry | 2d 3-body | `python main.py --dynamics nbody_2d_3b --group learn` |
| HNN | - | 2d 3-body | `python main.py --dynamics nbody_2d_3b --vi` |
| HNN | Oracle SE(2) symmetry | 2d 3-body | `python main.py --dynamics nbody_2d_3b --group se2 --vi` |
| HNN | Learned symmetry | 2d 3-body | `python main.py --dynamics nbody_2d_3b --group learn --vi` |

### Supported dynamical systems

The following dynamical systems are currently supported:

| Phase / Pixels | Dynamics | Symmetry | Availability of plotting code | Information
|---|---|---|---|---|
|Phase| Harmonic Oscillator | `--dynamics harmonic-oscillator` | Trajectory and energy plots supported. | Standard undampened harmonic oscillator. 
|Phase| N-body (e.g. 3-dimensional, 5 bodies) | `--dynamics nbody_3d_5b` | Trajectory plotting supported in 2d and 3d. | Supports arbitrary dimension and number of bodies `nbody_*d_*b`. 

New dynamical systems can easily be added by inheriting the `Dynamics` class in `dynamics.py`, which only requires specifying the Hamiltonian `dynamics.H()` of the system. Optionally, plotting functions can be added to this class. See implementations of existing dynamical systems as reference.

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
# noethers-razor
