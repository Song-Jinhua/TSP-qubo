import os
import sys
import time
import torch as th
import math
from graph_utils import load_graph_list, GraphList
from graph_utils import build_adjacency_bool, build_adjacency_indies, obtain_num_nodes
from graph_utils import update_xs_by_vs, gpu_info_str, evolutionary_replacement

TEN = th.Tensor


class SimulatorMaxcut:
    def __init__(self, sim_name: str = 'tsp', graph_list: GraphList = (), device=th.device('cpu')):
        self.device = device
        self.sim_name = sim_name
        self.int_type = th.long
        self.num_nodes = obtain_num_nodes(graph_list)  # Determine the number of nodes
        self.graphlist = graph_list
        self.if_maximize = False

        '''Create the QUBO matrix'''
        self.qubo_matrix = self.build_qubo_matrix().to(device)

    def build_qubo_matrix(self):
        '''Create the QUBO matrix for the TSP problem'''
        n = self.num_nodes
        qubo_matrix = th.zeros((n * n, n * n), dtype=th.float32)

        # Constraints that each node appears exactly once in the cycle
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if j != k:
                        qubo_matrix[i * n + j, i * n + k] += 2
                    if i != k:
                        qubo_matrix[i * n + j, k * n + j] += 2
                    qubo_matrix[i * n + j, i * n + j] -= 4

        # Objective function to minimize the Hamiltonian based on the given distances
        for (u, v, w) in self.graphlist:
            for j in range(n):
                k = (j + 1) % n  # Ensure it wraps around to create a cycle
                qubo_matrix[u * n + j, v * n + k] += w
                qubo_matrix[v * n + j, u * n + k] += w

        return qubo_matrix

    # def calculate_obj_values(self, xs: th.Tensor, if_sum: bool = True) -> th.Tensor:
    #     '''Calculate the objective values for a given solution using vectorized operations'''
    #     num_sims = xs.shape[0]  # Number of simulations/environments
    #     xs = xs.view(num_sims, -1)  # Flatten the solution matrices into vectors
    #
    #     # Calculate the objective values based on the QUBO matrix
    #     values = th.einsum('bi,ij,bj->b', xs.float(), self.qubo_matrix, xs)
    #
    #     if if_sum:
    #         values = values.sum(1)  # Sum the values if required
    #
    #     return values
    def calculate_obj_values(self, xs: th.Tensor, if_sum: bool = True) -> th.Tensor:
        '''Calculate the objective values for the TSP problem using the QUBO matrix'''
        num_sims = xs.shape[0]  # Number of simulations/environments
        num_nodes_squared = self.num_nodes * self.num_nodes

        # Ensure xs is flattened (if needed) and converted to float if not already
        xs_flat = xs.view(num_sims, num_nodes_squared).float()

        # Calculate the objective values based on the QUBO matrix
        # This performs the quadratic form: xs.T * Q * xs for each simulation
        values = th.einsum('bi,ij,bj->b', xs_flat, self.qubo_matrix, xs_flat)

        # At this point, values is a 1D tensor of shape (num_sims,), where each element is a scalar objective value
        return values

    def calculate_obj_values_for_loop(self, xs: th.Tensor, if_sum: bool = True) -> th.Tensor:
        '''Calculate the objective values for the TSP problem using the QUBO matrix'''
        num_sims = xs.shape[0]  # Number of simulations/environments
        num_nodes_squared = self.num_nodes * self.num_nodes

        # Ensure xs is flattened (if needed) and converted to float if not already
        xs_flat = xs.view(num_sims, num_nodes_squared).float()

        # Calculate the objective values based on the QUBO matrix
        # This performs the quadratic form: xs.T * Q * xs for each simulation
        values = th.einsum('bi,ij,bj->b', xs_flat, self.qubo_matrix, xs_flat)

        # At this point, values is a 1D tensor of shape (num_sims,), where each element is a scalar objective value
        return values

    #
    # def calculate_obj_values(self, xs: TEN, if_sum: bool = True) -> TEN:
    #     '''Calculate the objective values for a given solution'''
    #     xs = xs.to(self.device).flatten()
    #     values = xs @ self.qubo_matrix @ xs.t()
    #     if if_sum:
    #         values = values.sum(1)
    #     return values

    def generate_xs_randomly(self, num_sims):
        '''Generate random solutions and flatten them'''
        xs = th.randint(0, 2, size=(num_sims, self.num_nodes, self.num_nodes), dtype=th.bool, device=self.device)
        xs_flat = xs.view(num_sims, -1)  # Flatten xs into a 2D tensor with shape (num_sims, num_nodes * num_nodes)
        # print(f"simulator: generate_xs_randomly-- xs shape {xs.shape}, xs_flat shape {xs_flat.shape}")
        return xs_flat

    def local_search_inplace(self, good_xs: th.Tensor, good_vs: th.Tensor,
                             num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):
        '''Perform local search to find better solutions'''
        vs_raw = self.calculate_obj_values_for_loop(good_xs, if_sum=False)
        good_vs = vs_raw.sum(dim=1).long() if good_vs.shape == () else good_vs.long()
        rd_std = th.ones_like(vs_raw, dtype=th.float32) * noise_std
        spin_rand = vs_raw + th.randn_like(vs_raw, dtype=th.float32) * rd_std
        thresh = th.kthvalue(spin_rand, k=self.num_nodes * self.num_nodes - num_spin, dim=1)[0][:, None]

        for _ in range(num_iters):
            spin_rand = vs_raw + th.randn_like(vs_raw, dtype=th.float32) * rd_std
            spin_mask = spin_rand.gt(thresh)

            xs = good_xs.clone()
            xs[spin_mask] = th.logical_not(xs[spin_mask])
            vs = self.calculate_obj_values(xs)

            update_xs_by_vs(good_xs, good_vs, xs, vs, if_maximize=False)

        return good_xs, good_vs
















