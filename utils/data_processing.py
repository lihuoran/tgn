import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class Data:
    src_ids: np.ndarray
    dst_ids: np.ndarray
    timestamps: np.ndarray
    edge_idxes: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.n_interactions = len(self.src_ids)  # TODO: ?
        self.unique_nodes = set(self.src_ids) | set(self.dst_ids)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, use_validation=False):
    # Load data and train val test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    random.seed(2020)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(
    dataset_name: str,
    different_new_nodes_between_val_and_test: bool = False,
    randomize_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Data, Data, Data, Data, Data, Data]:
    # Load data and train val test split
    df: pd.DataFrame = pd.read_csv(f'./data/ml_{dataset_name}.csv')  # Meta info
    edge_features: np.ndarray = np.load(f'./data/ml_{dataset_name}.npy')  # Edge feature matrix
    node_features: np.ndarray = np.load(f'./data/ml_{dataset_name}_node.npy')  # Random node feature matrix

    if randomize_features:
        # Resample node features, randomly
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

    # Partition data into training / validation / testing set (70:15:15).
    val_time, test_time = list(np.quantile(df.ts, [0.70, 0.85]))

    src_ids: np.ndarray = df.u.values
    dst_ids: np.ndarray = df.i.values
    edge_idxes: np.ndarray = df.idx.values
    labels: np.ndarray = df.label.values
    timestamps: np.ndarray = df.ts.values

    full_data = Data(src_ids, dst_ids, timestamps, edge_idxes, labels)

    random.seed(2020)

    node_set = full_data.unique_nodes
    n_total_unique_nodes = full_data.n_unique_nodes

    # Compute nodes which appear at test time
    test_node_set = set(src_ids[timestamps > val_time]) | set(dst_ids[timestamps > val_time])

    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_src_mask = df.u.map(lambda x: x in new_test_node_set).values
    new_test_dst_mask = df.i.map(lambda x: x in new_test_node_set).values

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_src_mask, ~new_test_dst_mask)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    train_data = Data(
        src_ids[train_mask], dst_ids[train_mask],
        timestamps[train_mask], edge_idxes[train_mask], labels[train_mask],
    )

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
            [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(src_ids, dst_ids)]
        )
        edge_contains_new_test_node_mask = np.array(
            [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(src_ids, dst_ids)]
        )
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_data.src_ids) | set(train_data.dst_ids)
        assert len(train_node_set & new_test_node_set) == 0  # Should has no overlaps

        new_node_set = node_set - train_node_set  # TODO: why use new_node_set here?
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(src_ids, dst_ids)]
        )
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # Validation and test with all edges
    val_data = Data(
        src_ids[val_mask], dst_ids[val_mask],
        timestamps[val_mask], edge_idxes[val_mask], labels[val_mask],
    )
    test_data = Data(
        src_ids[test_mask], dst_ids[test_mask],
        timestamps[test_mask], edge_idxes[test_mask], labels[test_mask]
    )

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(
        src_ids[new_node_val_mask], dst_ids[new_node_val_mask],
        timestamps[new_node_val_mask], edge_idxes[new_node_val_mask], labels[new_node_val_mask]
    )
    new_node_test_data = Data(
        src_ids[new_node_test_mask], dst_ids[new_node_test_mask],
        timestamps[new_node_test_mask], edge_idxes[new_node_test_mask], labels[new_node_test_mask]
    )

    print(
        f"The dataset has {full_data.n_interactions} interactions, "
        f"involving {full_data.n_unique_nodes} different nodes"
    )
    print(
        f"The training dataset has {train_data.n_interactions} interactions, "
        f"involving {train_data.n_unique_nodes} different nodes"
    )
    print(
        f"The validation dataset has {val_data.n_interactions} interactions, "
        f"involving {val_data.n_unique_nodes} different nodes")
    print(
        f"The test dataset has {test_data.n_interactions} interactions, "
        f"involving {test_data.n_unique_nodes} different nodes")
    print(
        f"The new node validation dataset has {new_node_val_data.n_interactions} interactions, "
        f"involving {new_node_val_data.n_unique_nodes} different nodes")
    print(
        f"The new node test dataset has {new_node_test_data.n_interactions} interactions, "
        f"involving {new_node_test_data.n_unique_nodes} different nodes")
    print(
        "f{len(new_test_node_set)} nodes were used for the inductive testing, i.e. are never seen during training"
    )

    return (
        node_features, edge_features,
        full_data, train_data, val_data, test_data,
        new_node_val_data, new_node_test_data
    )


def compute_time_statistics(src_ids: np.ndarray, dst_ids: np.ndarray, timestamps: np.ndarray) -> tuple:
    last_ts_src = dict()
    last_ts_dst = dict()
    all_time_diffs_src = []
    all_time_diffs_dst = []
    for k in range(len(src_ids)):
        all_time_diffs_src.append(timestamps[k] - last_ts_src.get(src_ids[k], 0))  # TODO: how about the first diff?
        all_time_diffs_dst.append(timestamps[k] - last_ts_dst.get(dst_ids[k], 0))  # TODO: how about the first diff?
        last_ts_src[src_ids[k]] = last_ts_dst[dst_ids[k]] = timestamps[k]

    assert len(all_time_diffs_src) == len(src_ids)
    assert len(all_time_diffs_dst) == len(src_ids)
    mean_time_shift_src = np.mean(all_time_diffs_src)
    std_time_shift_src = np.std(all_time_diffs_src)
    mean_time_shift_dst = np.mean(all_time_diffs_dst)
    std_time_shift_dst = np.std(all_time_diffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
