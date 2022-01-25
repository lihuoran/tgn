from abc import abstractmethod
from typing import Any, List

from torch import nn
import torch

from modules.memory import Memory


class MemoryUpdater(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    @abstractmethod
    def update_memory(
        self,
        unique_node_ids: List[int],
        unique_messages: torch.Tensor,
        timestamps: torch.Tensor
    ) -> None:
        raise NotImplementedError


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory: Memory, message_dimension: int, memory_dimension: int, device: torch.device) -> None:
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(
        self,
        unique_node_ids: List[int],
        unique_messages: torch.Tensor,
        timestamps: torch.Tensor
    ) -> None:
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
            "Trying to update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps

        updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(
        self,
        unique_node_ids: List[int],
        unique_messages: torch.Tensor,
        timestamps: torch.Tensor
    ) -> tuple:
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
            "Trying to update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory: Memory, message_dimension: int, memory_dimension: int, device: torch.device) -> None:
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.GRUCell(input_size=message_dimension, hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory: Memory, message_dimension: int, memory_dimension: int, device: torch.device) -> None:
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.RNNCell(input_size=message_dimension, hidden_size=memory_dimension)


def get_memory_updater(
    module_type: str, memory: Memory, message_dimension: int, memory_dimension: int, device: torch.device
) -> MemoryUpdater:
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
