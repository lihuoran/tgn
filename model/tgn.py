import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from model.time_encoding import TimeEncode
from modules.embedding_module import EmbeddingModule, get_embedding_module
from modules.memory import Memory
from modules.memory_updater import get_memory_updater
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from utils.utils import MergeLayer, Message, NeighborFinder


class TGN(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(
        self,
        neighbor_finder: NeighborFinder,
        node_features: np.ndarray, edge_features: np.ndarray,
        device: torch.device,
        n_layers: int = 2, n_heads: int = 2, dropout: float = 0.1, use_memory: bool = False,
        memory_update_at_start: bool = True, message_dimension: int = 100, memory_dimension: int = 500,
        embedding_module_type: str = "graph_attention", message_function: str = "mlp",
        mean_time_shift_src: float = 0.0, std_time_shift_src: float = 1.0,
        mean_time_shift_dst: float = 0.0, std_time_shift_dst: float = 1.0,
        n_neighbors: int = None, aggregator_type: str = "last", memory_updater_type: str = "gru",
        use_destination_embedding_in_message: bool = False, use_source_embedding_in_message: bool = False,
        dyrep: bool = False,
    ) -> None:
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_dst_emb_in_message = use_destination_embedding_in_message
        self.use_src_emb_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self._time_encoder = TimeEncode(dimension=self.n_node_features)  # TODO: dive into TimeEncode
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + self._time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
            self.memory = Memory(
                n_nodes=self.n_nodes,
                memory_dimension=self.memory_dimension,
                input_dimension=message_dimension,
                message_dimension=message_dimension,
                device=device
            )
            self._message_aggregator = get_message_aggregator(
                aggregator_type=aggregator_type,
                device=device
            )
            self._message_function = get_message_function(
                module_type=message_function,
                raw_message_dimension=raw_message_dimension,
                message_dimension=message_dimension
            )
            self._memory_updater = get_memory_updater(
                module_type=memory_updater_type,
                memory=self.memory,
                message_dimension=message_dimension,
                memory_dimension=self.memory_dimension,
                device=device
            )

        self.embedding_module_type = embedding_module_type

        self.embedding_module: EmbeddingModule = get_embedding_module(
            module_type=embedding_module_type,
            node_features=self.node_raw_features,
            edge_features=self.edge_raw_features,
            memory=self.memory,
            neighbor_finder=self.neighbor_finder,
            time_encoder=self._time_encoder,
            n_layers=self.n_layers,
            n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,
            n_time_features=self.n_node_features,
            embedding_dimension=self.embedding_dimension,
            device=self.device,
            n_heads=n_heads, dropout=dropout,
            use_memory=use_memory,
            n_neighbors=self.n_neighbors
        )

        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features, self.n_node_features, 1)

    def compute_temporal_embeddings(
        self,
        src_ids: np.ndarray, dst_ids: np.ndarray,
        neg_ids: np.ndarray, edge_times: np.ndarray, edge_idxes: np.ndarray,
        n_neighbors: int = 20,
    ):
        n_samples = len(src_ids)
        nodes = np.concatenate([src_ids, dst_ids, neg_ids])
        positives = np.concatenate([src_ids, dst_ids])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches
                memory, last_update = self._get_updated_memory(list(range(self.n_nodes)), self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            # Compute differences between the time the memory of a node was last updated,
            # and the time for which we want to compute the embedding of a node
            src_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[src_ids].long()
            src_time_diffs = (src_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src  # Gaussian normalize
            dst_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[dst_ids].long()
            dst_time_diffs = (dst_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst  # Gaussian normalize
            neg_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[neg_ids].long()
            neg_time_diffs = (neg_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst  # Gaussian normalize

            time_diffs = torch.cat([src_time_diffs, dst_time_diffs, neg_time_diffs], dim=0)

        # Compute the embeddings using the embedding module
        node_embedding = self.embedding_module.compute_embedding(  # Embedding equation in page 4
            memory=memory,
            src_ids=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs
        )

        src_emb = node_embedding[:n_samples]
        dst_emb = node_embedding[n_samples: 2 * n_samples]
        neg_emb = node_embedding[2 * n_samples:]

        if self.use_memory:
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                self._update_memory(positives, self.memory.messages)

                assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
                    "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)

            uniq_src_ids, src_id_to_messages = self._get_raw_messages(
                src_ids, src_emb, dst_ids, dst_emb, edge_times, edge_idxes
            )
            uniq_dst_ids, dst_id_to_messages = self._get_raw_messages(
                dst_ids, dst_emb, src_ids, src_emb, edge_times, edge_idxes
            )
            if self.memory_update_at_start:
                self.memory.store_raw_messages(uniq_src_ids, src_id_to_messages)
                self.memory.store_raw_messages(uniq_dst_ids, dst_id_to_messages)
            else:
                self._update_memory(uniq_src_ids, src_id_to_messages)
                self._update_memory(uniq_dst_ids, dst_id_to_messages)

            if self.dyrep:  # TODO: ?
                src_emb = memory[src_ids]
                dst_emb = memory[dst_ids]
                neg_emb = memory[neg_ids]

        return src_emb, dst_emb, neg_emb

    def compute_edge_probabilities(
        self,
        src_ids: np.ndarray, dst_ids: np.ndarray,
        neg_ids: np.ndarray, edge_times: np.ndarray, edge_idxes: np.ndarray,
        n_neighbors: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute probabilities for edges between sources and destination and between sources and
        # negatives by first computing temporal embeddings using the TGN encoder and then feeding them
        # into the MLP decoder.
        # :param destination_nodes [batch_size]: destination ids
        # :param negative_nodes [batch_size]: ids of negative sampled destination
        # :param edge_times [batch_size]: timestamp of interaction
        # :param edge_idxes [batch_size]: index of interaction
        # :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        # layer
        # :return: Probabilities for both the positive and negative edges
        n_samples = len(src_ids)
        src_emb, dst_emb, neg_emb = self.compute_temporal_embeddings(
            src_ids, dst_ids, neg_ids, edge_times, edge_idxes, n_neighbors
        )

        score = self.affinity_score(
            torch.cat([src_emb, src_emb], dim=0),
            torch.cat([dst_emb, neg_emb])
        ).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()

    def _update_memory(self, nodes: np.ndarray, messages: dict) -> None:
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = self._message_aggregator.aggregate(nodes, messages)

        if len(unique_nodes) > 0:
            unique_messages = self._message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self._memory_updater.update_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)

    def _get_updated_memory(self, nodes: List[int], messages: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # Aggregate messages for the same nodes
        # E.g., last message / mean message
        unique_nodes, unique_messages, unique_timestamps = self._message_aggregator.aggregate(nodes, messages)

        if len(unique_nodes) > 0:
            # E.g., identity / MLP
            unique_messages = self._message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self._memory_updater.get_updated_memory(  # E.g., GRU / RNN
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )

        return updated_memory, updated_last_update

    def _get_raw_messages(
        self,
        src_ids: np.ndarray, src_emb: torch.Tensor,
        dst_ids: np.ndarray, dst_emb: torch.Tensor,
        edge_times: np.ndarray, edge_idxes: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, List[Message]]]:
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxes]

        src_memory = self.memory.get_memory(src_ids) if not self.use_src_emb_in_message else src_emb
        dst_memory = self.memory.get_memory(dst_ids) if not self.use_dst_emb_in_message else dst_emb

        source_time_delta = edge_times - self.memory.last_update[src_ids]
        source_time_delta_encoding = self._time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(src_ids), -1)

        source_message = torch.cat(
            [src_memory, dst_memory, edge_features, source_time_delta_encoding],
            dim=1
        )
        messages: Dict[int, List[Message]] = defaultdict(list)
        unique_sources = np.unique(src_ids)

        for i in range(len(src_ids)):
            messages[src_ids[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
