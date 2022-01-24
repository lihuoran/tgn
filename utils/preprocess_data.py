import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def preprocess(data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_path) as f:
        next(f)  # Skip the header line
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])  # User id
            i = int(e[1])  # Item id
            ts = float(e[2])  # Timestamp
            label = float(e[3])  # Binary label, 0 or 1
            feat = np.array([float(x) for x in e[4:]])  # Feature vector

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            feat_l.append(feat)

    feature_meta_df = pd.DataFrame({
        'u': u_list, 'i': i_list,
        'ts': ts_list, 'label': label_list, 'idx': idx_list,
    })
    feature_array = np.array(feat_l)
    return feature_meta_df, feature_array


def reindex(df: pd.DataFrame, bipartite: bool = True) -> pd.DataFrame:
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        # User id range: [0, upper_u)
        # Item id range: [upper_u, upper_u + num_item)
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u
        new_df.i = new_i

    # 0-indexed => 1-indexed
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    return new_df


def run(data_path: str, bipartite: bool = True) -> None:
    Path("data/").mkdir(parents=True, exist_ok=True)
    csv_path = f'./data/{data_path}.csv'
    ml_csv_path = f'./data/ml_{data_path}.csv'
    ml_npy_path = f'./data/ml_{data_path}.npy'
    ml_node_npy_path = f'./data/ml_{data_path}_node.npy'

    df, feat = preprocess(csv_path)
    new_df = reindex(df, bipartite)

    empty = np.zeros((1, feat.shape[1]))  # Empty feature vector with shape (1, nfeature)
    feat = np.vstack([empty, feat])  # Expanded feature matrix with shape (nentry + 1, nfeature)

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))  # Random node feature matrix. TODO: hardcoded nfeature?

    new_df.to_csv(ml_csv_path)
    np.save(ml_npy_path, feat)
    np.save(ml_node_npy_path, rand_feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
    parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='wikipedia')
    parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
    args = parser.parse_args()

    run(args.data, bipartite=args.bipartite)
