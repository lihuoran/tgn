import math
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from model.tgn import TGN
from utils.data_processing import Data
from utils.utils import RandEdgeSampler


def eval_edge_prediction(
    model: TGN,
    negative_edge_sampler: RandEdgeSampler,
    data: Data,
    n_neighbors: int,
    batch_size: int = 200,
) -> Tuple[float, float]:
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.src_ids)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            src_ids = data.src_ids[s_idx:e_idx]
            dst_ids = data.dst_ids[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxes_batch = data.edge_idxes[s_idx: e_idx]

            size = len(src_ids)
            _, neg_ids = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(
                src_ids, dst_ids, neg_ids, timestamps_batch, edge_idxes_batch, n_neighbors
            )

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return float(np.mean(val_ap)), float(np.mean(val_auc))


def eval_node_classification(
    tgn: TGN,
    decoder: nn.Module,
    data: Data,
    edge_idxes: np.ndarray,
    batch_size: int,
    n_neighbors: int,
) -> float:
    pred_prob = np.zeros(len(data.src_ids))
    num_instance = len(data.src_ids)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            src_ids = data.src_ids[s_idx: e_idx]
            dst_ids = data.dst_ids[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxes_batch = edge_idxes[s_idx: e_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
                src_ids, dst_ids, dst_ids, timestamps_batch, edge_idxes_batch, n_neighbors
            )
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
