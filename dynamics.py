import re

from abc import abstractmethod
from functools import partial

import jax
import jax.numpy as jnp

from network import batch_mlp, sample_weights

import matplotlib
import matplotlib.pyplot as plt


class Dynamics(object): 
    def __init__(self, cdim):
        self.cdim = cdim
        self.pdim = cdim * 2

    @abstractmethod
    def H(self, x): 
        """ Return scalar energy given phase space location x of shape (pdim,) """
        pass

    def initial_phase(self, key, q_scale=1.0, p_scale=1.0, q_shift=0.0, p_shift=0.0, q_trans=0.0, p_trans=0.0):
        """ Return initial location of shape (2, cdim). """
        subkey, key = jax.random.split(key)
        x_start = jax.random.normal(subkey, shape=(2, self.cdim))

        subkey, key = jax.random.split(key)
        q_shift = jax.random.normal(subkey) * q_shift
        subkey, key = jax.random.split(key)
        p_shift = jax.random.normal(subkey) * p_shift

        qp_scale = jnp.array([q_scale, p_scale]).reshape(2, 1)
        qp_shift = jnp.array([q_shift, p_shift]).reshape(2, 1)
        qp_trans = jnp.array([q_trans, p_trans]).reshape(2, 1)
        x_start = x_start * qp_scale + qp_shift + qp_trans
        
        return x_start

    def plot_trajectory(self, trajectory, t_span, ax, transparent=False):
        raise NotImplementedError(f"No trajectory plotting available for selected dynamical system.")

    def plot_H(self, ax):
        raise NotImplementedError(f"No energy plotting available for selected dynamical system.")

    def canonicaliser(self, group):
        raise NotImplementedError(f"Canonicalisation of '{group}' not implemented for selected dynamical system")

    def conserved_functions(self, group):
        if group == '':
            return []
        else:
            raise NotImplementedError(f"Conservation of '{group}' not implemented for selected dynamical system")


class Nbody(Dynamics):
    def __init__(self, dim, n_bodies, masses=1.0):
        cdim = dim * n_bodies
        super().__init__(cdim)

        self.dim = dim
        self.n_bodies = n_bodies

        self.masses = masses

    def H(self, x, eps=1.0):
        assert len(x) == self.pdim, f"x does not have correct shape of {self.pdim}. Got x of shape {x.shape}."

        masses = jnp.array([self.masses for _ in range(self.n_bodies)]) if type(self.masses) == float else self.masses
        
        q, p = x.reshape(2, self.n_bodies, self.dim)

        cross_masses = masses ** 2

        H_kinetic = jnp.sum((jnp.linalg.norm(p, axis=1) ** 2) / (2 * masses))

        q_dists = q.reshape(self.n_bodies, 1, self.dim) - q.reshape(1, self.n_bodies, self.dim) # (n_bodies, n_bodies, dim)
        q_quads = jnp.sum(q_dists ** 2, axis=2) # (n_bodies, n_bodies)
        masses_outer = jnp.outer(masses, masses) # (n_bodies, n_bodies)
        H_potential = -jnp.sum(jnp.tril(masses_outer / jnp.sqrt(q_quads + (eps ** 2)), -1))

        H = H_kinetic + H_potential

        return H

    def plot_phase(self, x, ax, xlim=3, ylim=3, quiver_scale=1.0, quiver_width=1.0, alpha=1.0):
        q, p = x.reshape(2, self.n_bodies, self.dim)

        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:green', 'tab:brown', 'tab:pink']
        
        for object_i, (q_item, p_item) in enumerate(zip(q, p)):
            color = colors[object_i % len(colors)]

            ax.scatter(q_item[0], q_item[1], color=color, alpha=alpha)
            ax.quiver(q_item[0], q_item[1], -p_item[0], -p_item[1], color=color, scale=quiver_scale, width=quiver_width, alpha=alpha, angles='xy')

        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        # ax.set_xticks([])
        # ax.set_yticks([])

        ax.set_aspect('equal')


    def plot_trajectory(self, trajectory, t_span, ax, n_line_segments=500, xlim=3, ylim=3, H=None, JH=None, alpha=1.0, transparent=False):
        """ Plot 2d trajectory

            Input:
                trajectory: (T, 2, n_bodies, dim)
        """

        if self.dim != 2:
            raise NotImplementedError(f"N-body system plotting currently only supported for 2d systems")
        
        T = trajectory.shape[0]

        trajectory = trajectory.reshape(T, 2, self.n_bodies, self.dim)

        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:green', 'tab:brown', 'tab:pink']
        if transparent:
            colors = [tuple([x + (1 - x) * 0.6 for x in matplotlib.colors.to_rgb(color)]) for color in colors]
        
        for object_i in range(self.n_bodies):
            # color
            color = colors[object_i % len(colors)]

            points = trajectory.reshape(T, 2, self.n_bodies, self.dim)[:, 0, object_i]

            points_x, points_y = points[:, 0], points[:, 1]

            if transparent:
                # draw dots
                ax.plot(points_x, points_y, '-', linewidth=1, color=color, alpha=alpha)
            else:
                # draw line
                ax.plot(points_x, points_y, '-', linewidth=1, color=color, alpha=alpha)

                # draw line
                ax.plot(points_x[-1:], points_y[-1:], '.', markersize=8, color=color, alpha=alpha)

        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal')

    def canonicaliser(self, group):
        def canonicalise_t2(x):
            assert len(x.shape) == 1, f"Assumes dimension (P)... Got {x.shape}"

            x_view = x.reshape(2, self.n_bodies, self.dim)
            
            mask = jnp.array([1, 0]).reshape(-1, 1, 1)
            x_view = x_view - mask * x_view.mean(1, keepdims=True)
            
            return x_view.reshape(x.shape)

        return canonicalise_t2

    def init_affine_as_subgroup(self, subgroup, std=0.0, key=None):
        W = jnp.zeros(shape=(self.pdim, self.pdim))
        b = jnp.zeros(shape=(self.pdim,))

        if 't_' in subgroup:
            _, d = subgroup.split('_')
            d = int(d)

            b = b.reshape(2, self.n_bodies, self.dim).at[1, :, d].set(1.0).reshape(*b.shape)
        elif 'rcom_' in subgroup:
            _, d1, d2 = subgroup.split('_')
            d1, d2 = int(d1), int(d2)

            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[0, :, d1, 1, :, d2].set(1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[0, :, d2, 1, :, d1].set(-1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[1, :, d1, 0, :, d2].set(-1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[1, :, d2, 0, :, d1].set(1.0).reshape(*W.shape)
        elif 'rabs_' in subgroup:
            _, d1, d2 = subgroup.split('_')
            d1, d2 = int(d1), int(d2)

            index = jnp.arange(0, self.n_bodies)

            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[0, index, d1, 1, index, d2].set(1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[0, index, d2, 1, index, d1].set(-1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[1, index, d1, 0, index, d2].set(-1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[1, index, d2, 0, index, d1].set(1.0).reshape(*W.shape)
        elif 'p_' in subgroup:
            _, d = subgroup.split('_')
            d = int(d)

            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[1, :, d, 1, :, d].set(1.0).reshape(*W.shape)
        elif 'q_' in subgroup:
            _, d1, d2 = subgroup.split('_')
            d1, d2 = int(d1), int(d2)

            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[1, :, d1, 1, :, d2].set(1.0).reshape(*W.shape)

        elif subgroup == 'translate_x':
            b = b.reshape(2, self.n_bodies, self.dim).at[1, :, 0].set(1.0).reshape(*b.shape)
        elif subgroup == 'translate_y':
            assert self.dim > 1, f"Translation in y-axis only exists in dim > 1"
            b = b.reshape(2, self.n_bodies, self.dim).at[1, :, 1].set(1.0).reshape(*b.shape)
        elif subgroup == 'translate_z':
            assert self.dim > 2, f"Translation in z-axis only exists in dim > 2"
            b = b.reshape(2, self.n_bodies, self.dim).at[1, :, 2].set(1.0).reshape(*b.shape)
        elif subgroup == 'rabs_0_1':
            assert self.dim == 2, f"Rotation currently only implemented in 2-dimensions"
            index = jnp.arange(0, self.n_bodies)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[0, index, 0, 1, index, 1].set(1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[0, index, 1, 1, index, 0].set(-1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[1, index, 0, 0, index, 1].set(-1.0).reshape(*W.shape)
            W = W.reshape(2, self.n_bodies, self.dim, 2, self.n_bodies, self.dim).at[1, index, 1, 0, index, 0].set(1.0).reshape(*W.shape)
            jax.debug.print("W: {x}", x=W)
        elif subgroup == '':
            W_key, b_key, key = jax.random.split(key, 3)
            W = jax.random.normal(key=W_key, shape=(self.pdim, self.pdim)) * std
            b = jax.random.normal(key=b_key, shape=(self.pdim,)) * std
        else:
            raise NotImplementedError(f"Unknown group. Got: {subgroup}.")

        return [{'W': W, 'b': b}]

    def conserved_functions(self, group):
        def mlp(cypers, x):
            key = None
            vi = False
            deterministic = True

            weights = sample_weights(cypers, False, use_mean=True)
            return batch_mlp(weights, 'elu', x.reshape(1, -1)).squeeze()

        def quadratic(cyper, x):
            assert len(cyper) == 1, f"Error. Assuming quadratic function is implemented as 1 layer"
            W, b = cyper[0]['W'], cyper[0]['b']

            out = 0.5 * x.reshape(1, -1) @ W @ x.reshape(-1, 1) + jnp.dot(b, x)

            return jnp.mean(out) # Todo fix
            #raise NotImplementedError(f"Quadratic conserved quantity not yet implemented.")

        def conserved_momentum_trans(NOT_LEARNED, x, trans_axis=0):
            return x.reshape(2, self.n_bodies, self.dim)[1, :, trans_axis].sum(-1)

        def conserved_momentum_angular(NOT_LEARNED, x, rotary_axis=0):
            x_view = x.reshape(2, self.n_bodies, self.dim)

            if self.dim == 2: # in 2 dims, add 0 to allow same cross-product calculation as in 3 dims.
                added_zero = jnp.zeros((*x_view.shape[:-1], 1))
                x_view = jnp.concatenate((x_view, added_zero), -1) # (2, self.n_bodies, 3)

                angular_momentum = jnp.cross(x_view[0], x_view[1]).sum(0) # (3)
                angular_momentum = angular_momentum[-1:] # (1,)
            elif self.dim == 3:
                angular_momentum = jnp.cross(x_view[0], x_view[1]).sum(0) # (3)
            else:
                raise NotImplementedError(f"Angular momentum only specified for dim=2 and dim=3. Got dim={self.dim}") 

            return angular_momentum[rotary_axis] * jnp.pi

        if 'quadratic' in group:
            if 'learn' in group:
                group_dim = int(re.findall(r'\d+', group)[0])
            elif 't2' in group:
                group_dim = 2
            elif 'so2' in group:
                group_dim = 1
            elif 'se2' in group:
                group_dim = 3
            elif 'sen' in group:
                group_dim = int(2 * self.dim + 3 * ((self.dim - 1) * self.dim / 2))

            return [quadratic for _ in range(group_dim)]
        else:
            if group == 't2':
                return [partial(conserved_momentum_trans, trans_axis=0),
                        partial(conserved_momentum_trans, trans_axis=1)]
            elif group == 'so2':
                return [conserved_momentum_angular]
            elif group == 'se2':
                return [partial(conserved_momentum_trans, trans_axis=0),
                        partial(conserved_momentum_trans, trans_axis=1),
                        conserved_momentum_angular]
            elif group == 'se3':
                return [partial(conserved_momentum_trans, trans_axis=0),
                        partial(conserved_momentum_trans, trans_axis=1),
                        partial(conserved_momentum_trans, trans_axis=2),
                        partial(conserved_momentum_angular, rotary_axis=0),
                        partial(conserved_momentum_angular, rotary_axis=1),
                        partial(conserved_momentum_angular, rotary_axis=2),
                        ]
            elif group == 'learn':
                group_dim = int(re.findall(r'\d+', group)[0])
                return [mlp for _ in range(group_dim)]
            elif group == '':
                return []

        raise NotImplementedError(f"Group not supported. Got: {group}.")

class HarmonicOscillator(Dynamics):
    def __init__(self, omega=1.0):
        cdim = 1
        super().__init__(cdim)

        self.omega = omega

    def H(self, x, eps=1.0):   
        assert x.shape[0] == 2, f"x does have have correct shape: {x}"

        q, p = x

        pot_energy = 0.5 * (q ** 2)
        kin_energy = 0.5 * (p ** 2) * (self.omega ** 2)

        return pot_energy + kin_energy

    def init_affine_as_subgroup(self, group, std=0.0, key=None):
        W = jnp.zeros(shape=(self.pdim, self.pdim))
        b = jnp.zeros(shape=(self.pdim,))

        if group == 'rabs_0_1':
            W = W.at[jnp.diag_indices_from(W)].set(1.0)
        elif group == '':
            W_key, b_key, key = jax.random.split(key, 3)
            W = jax.random.normal(key=W_key, shape=(self.pdim, self.pdim)) * std
            b = jax.random.normal(key=b_key, shape=(self.pdim,)) * std
        else:
            raise NotImplementedError(f"Unknown group. Got: {group}.")

        return [{'W': W, 'b': b}]

    def conserved_functions(self, group):
        def mlp(cypers, x):
            key = None
            vi = False
            deterministic = True

            weights = sample_weights(cypers, False, use_mean=True)
            return batch_mlp(weights, 'elu', x.reshape(1, -1)).squeeze()

        def quadratic(cypers, x):
            raise NotImplementedError(f"Quadratic function not yet implemented")

        def conserved_phase(EMPTY, x):
            return self.H(x) * jnp.pi

        if group == 'so2':
            return [conserved_phase]
        elif 'quadratic' in group:
            if 'so2' in group:
                group_dim = 1
            else:
                group_dim = int(re.findall(r'\d+', group)[0])
            return [quadratic for _ in range(group_dim)]
        elif 'mlp' in group:
            if 'learn' in group:
                return [mlp for _ in range(group_dim)]
        elif group == '':
            return []

        raise NotImplementedError(f"Unknown group. Got {group}")

    def plot_trajectory(self, trajectory, t_span, ax, n_line_segments=500, ylim=3, H=None, JH=None, alpha=1.0, transparent=False):
        """ Plot 2d trajectory

            Input:
                trajectory: (T, pdim)
        """

        T = trajectory.shape[0]

        trajectory = trajectory.reshape(T, 2, 1)

        points_x, points_y = [], []

        # color
        color = 'black'

        time = t_span

        points = trajectory.reshape(T, 2)[:, 0]

        if transparent:
            # draw brighter
            ax.plot(time, points, '-', linewidth=2, color=color, alpha=alpha)
        else:
            # draw line
            ax.plot(time, points, '-', linewidth=2, markersize=5, color=color, alpha=alpha)

            # draw ball
            ax.plot([time[-1]], [points[-1]], '.', markersize=6, color=color, alpha=alpha)

        ax.set_xlim(0, t_span[-1] * 1.1)
        ax.set_ylim(-ylim, ylim)

    def plot_phase_energy(self, grid_energy, ax, lim=1.0, levels=None):
        """ Plot energy on phase space

            Input:
                energy grid: (H, W)
        """

        #ax.imshow(grid_energy, extent=(-lim, lim, -lim, lim))

        H, W = grid_energy.shape
        y = jnp.linspace(-lim, lim, H)
        x = jnp.linspace(-lim, lim, W)
        xx, yy = jnp.meshgrid(x, y)
        ax.contour(xx, yy, grid_energy, levels=levels)

        # ax.set_xlim(0, 1)
        # ax.set_ylim(-ylim, ylim)
        ax.set_xticks([-lim, 0, lim])
        ax.set_yticks([-lim, 0, lim])

        ax.set_aspect('equal')

    def plot_phase_energy_line(self, grid_energy, ax, lim=1.0):
        """ Plot energy on phase space in the line q=p

            Input:
                energy grid: (H, W)
        """

        H, W = grid_energy.shape

        assert H == W, f"Plotting in the line q=p requires square energy grid" 

        ax.plot(jnp.linspace(-lim, lim, H), jnp.diag(grid_energy))

        ax.set_xticks([-lim, 0, lim])
        ax.set_yticks([-lim, 0, lim])

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.set_aspect('equal')

    def plot_phase_trajectories(self, trajectories, ax, n_line_segments=500, lim=3, H=None, JH=None, alpha=1.0, transparent=False):
        """ Plot 2d trajectory

            Input:
                trajectories: (B, T, pdim)
        """

        B, T, pdim = trajectories.shape

        if pdim != 2:
            raise NotImplementedError(f"Can onnly plot phase trajectories for 2-dimensional phase spaces")

        for trajectory in trajectories:
            points_x, points_y = [], []

            # color
            color = 'black'

            time = jnp.linspace(0, 1, T)

            q_points = trajectory.reshape(T, 2)[:, 0]
            p_points = trajectory.reshape(T, 2)[:, 1]

            if transparent:
                # draw dots
                for t_i in range(T):
                    ax.scatter([q_points[t_i]], [p_points[t_i]], s=3, markersize=10, marker='.', color=color, alpha=alpha)
            else:
                # draw line
                ax.plot(q_points, p_points, '.-', markersize=10, color=color, alpha=0.5 * alpha)

                # # draw ball
                # ax.plot([q_points[-1]], [p_points[-1]], '>', markersize=5, color=color, alpha=alpha)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # ax.set_xticks([])
        # ax.set_yticks([])

        ax.set_xlabel('q')
        ax.set_ylabel('p')

        ax.set_aspect('equal')


class NHarmonicOscillator(Dynamics):
    def __init__(self, n_harmonics, omega=1.0):
        cdim = n_harmonics
        super().__init__(cdim)

        self.omega = omega

    def H(self, x, eps=1.0):   
        assert len(x) == self.pdim, f"x does not have correct shape of {self.pdim}. Got x of shape {x.shape}."
        q, p = x.reshape(2, self.cdim)

        pot_energy = 0.5 * jnp.sum(q ** 2)
        kin_energy = 0.5 * jnp.sum(p ** 2) * (self.omega ** 2)

        return pot_energy + kin_energy

    def init_affine_as_subgroup(self, group, std=0.0, key=None):
        W = jnp.zeros(shape=(self.pdim, self.pdim))
        b = jnp.zeros(shape=(self.pdim,))

        if group == 'rabs_0_1':
            W = W.at[jnp.diag_indices_from(W)].set(1.0)
        elif 'un' in group:
            """ Create a certain conserved quantity associated to group U(n) given by un_[generator_id] """
            ni = int(re.findall(r'\d+', group)[0])
            assert (ni >= 0) and (ni < (self.cdim ** 2)), f"{ni} not between 0 and n^2 ({self.cdim ** 2}), with n={self.cdim}."
            i, j = ni // self.cdim, ni % self.cdim
            assert (i >= 0) and (i < self.cdim ** 2), f"Expects 0 < i < n^2 ({self.cdim ** 2}). Got i={i}"
            assert (j >= 0) and (j < self.cdim ** 2), f"Expects 0 < j < n^2 ({self.cdim ** 2}). Got j={j}"

            if i < j: # R_ij
                W = W.reshape(2, self.cdim, 2, self.cdim).at[0, i, 1, j].set(1.0)
                W = W.reshape(2, self.cdim, 2, self.cdim).at[0, j, 1, i].set(-1.0)
                W = W.reshape(2 * self.cdim, 2 * self.cdim)
            elif j < i: # F_ij
                W = W.reshape(2, self.cdim, 2, self.cdim).at[0, i, 0, j].set(1.0)
                W = W.reshape(2, self.cdim, 2, self.cdim).at[1, i, 1, j].set(1.0)
                W = W.reshape(2 * self.cdim, 2 * self.cdim)
            elif i == j: # H_i
                W = W.reshape(2, self.cdim, 2, self.cdim).at[0, i, 0, i].set(1.0)
                W = W.reshape(2, self.cdim, 2, self.cdim).at[1, i, 1, i].set(1.0)
                W = W.reshape(2 * self.cdim, 2 * self.cdim)
            else:   
                raise NotImplementedError(f"Unexpected case.")

        elif group == '':
            W_key, b_key, key = jax.random.split(key, 3)
            W = jax.random.normal(key=W_key, shape=(self.pdim, self.pdim)) * std
            b = jax.random.normal(key=b_key, shape=(self.pdim,)) * std
        else:
            raise NotImplementedError(f"Unknown group. Got: {group}.")

        return [{'W': W, 'b': b}]

    def conserved_functions(self, group):
        def mlp(cypers, x):
            key = None
            vi = False
            deterministic = True

            weights = sample_weights(cypers, False, use_mean=True)
            return batch_mlp(weights, 'elu', x.reshape(1, -1)).squeeze()

        def quadratic(cypers, x):
            raise NotImplementedError(f"Quadratic function not yet implemented")

        def conserved_phase(EMPTY, x):
            return self.H(x) * jnp.pi

        if group == 'so2':
            return [conserved_phase]
        elif 'quadratic' in group:
            if 'so2' in group:
                group_dim = 1
            elif 'un' in group:
                group_dim = self.cdim ** 2
            else:
                group_dim = int(re.findall(r'\d+', group)[0])
            return [quadratic for _ in range(group_dim)]
        elif 'mlp' in group:
            if 'learn' in group:
                return [mlp for _ in range(group_dim)]
        elif group == '':
            return []

        raise NotImplementedError(f"Unknown group. Got {group}")

