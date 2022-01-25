from typing import Any, List, Tuple

import numpy as np
import torch

from .data_processing import Data


class MergeLayer(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, dim1: int, dim2: int, dim3: int, dim4: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(dim3, dim4)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MLP(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, dim: int, drop: float = 0.3) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
    def __init__(self, max_round: int = 3, higher_better: bool = True, tolerance: float = 1e-10) -> None:
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val: float) -> bool:
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val

        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    """
    Randomly sample edges (user-item pairs) in the given bipartite graph
    """
    def __init__(self, src_id_list: np.ndarray, dst_id_list: np.ndarray, seed: int = None) -> None:
        self._seed = None
        self._src_id_list = np.unique(src_id_list)
        self._dst_id_list = np.unique(dst_id_list)

        if seed is not None:
            self._seed = seed
            self._random_state = np.random.RandomState(self._seed)

    def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._seed is None:
            src_index = np.random.randint(0, len(self._src_id_list), size)
            dst_index = np.random.randint(0, len(self._dst_id_list), size)
        else:
            src_index = self._random_state.randint(0, len(self._src_id_list), size)
            dst_index = self._random_state.randint(0, len(self._dst_id_list), size)
        return self._src_id_list[src_index], self._dst_id_list[dst_index]

    def reset_random_state(self) -> None:
        self._random_state = np.random.RandomState(self._seed)


class NeighborFinder:
    def __init__(self, adj_list: List[list], uniform: bool = False, seed: int = None) -> None:
        self.node_to_neighbors = []
        self.node_to_edge_idxes = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighbors = sorted(neighbors, key=lambda x: x[2])  # Sort by timestamp, ascending
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbors]))
            self.node_to_edge_idxes.append(np.array([x[1] for x in sorted_neighbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighbors]))

        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx: int, cut_time: float) -> tuple:
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph.
        The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxes, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], \
            self.node_to_edge_idxes[src_idx][:i], \
            self.node_to_edge_timestamps[src_idx][:i]

    def get_temporal_neighbor(self, src_ids: List[int], timestamps: List[float], n_neighbors: int = 20) -> tuple:
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user
        in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(src_ids) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1

        # NB! All interactions described in these matrices are sorted in each row by time

        # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an
        # interaction happening before cut_time_l[i]
        neighbors = np.zeros((len(src_ids), tmp_n_neighbors)).astype(np.int32)

        # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and
        # item neighbors[i,j] happening before cut_time_l[i]
        edge_times = np.zeros((len(src_ids), tmp_n_neighbors)).astype(np.float32)

        # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i]
        # and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxes = np.zeros((len(src_ids), tmp_n_neighbors)).astype(np.int32)

        for i, (source_node, timestamp) in enumerate(zip(src_ids, timestamps)):
            # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node
            # happening before cut_time
            source_neighbors, source_edge_idxes, source_edge_times = self.find_before(source_node, timestamp)

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)
                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxes[i, :] = source_edge_idxes[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxes[i, :] = edge_idxes[i, :][pos]
                else:
                    # Take most recent interactions
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_edge_idxes = source_edge_idxes[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxes) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxes[i, n_neighbors - len(source_edge_idxes):] = source_edge_idxes

        return neighbors, edge_idxes, edge_times


def get_neighbor_finder(data: Data, uniform: bool, max_node_idx: int = None) -> NeighborFinder:
    max_node_idx = max(np.max(data.src_ids), np.max(data.dst_ids)) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]

    data_iter = zip(data.src_ids, data.dst_ids, data.edge_idxes, data.timestamps)
    for src_id, dst_id, edge_idx, timestamp in data_iter:
        adj_list[int(src_id)].append((dst_id, edge_idx, timestamp))
        adj_list[int(dst_id)].append((src_id, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)
