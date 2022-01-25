from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn

from utils.utils import Message


class Memory(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(
        self,
        n_nodes: int,
        memory_dimension: int,
        input_dimension: int,
        message_dimension: int = None,
        device: torch.device = "cpu",
        combination_method: str = "sum"
    ) -> None:
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device

        self.combination_method = combination_method

        self.messages: Dict[int, List[Message]] = defaultdict(list)
        self.__init_memory__()

    def __init_memory__(self) -> None:
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory = nn.Parameter(
            torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
            requires_grad=False
        )
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device), requires_grad=False)

        self.messages = defaultdict(list)

    def store_raw_messages(
        self, nodes: Union[List[int], np.ndarray], node_id_to_messages: Dict[int, List[Message]]
    ) -> None:
        for node in nodes:
            self.messages[node] += node_id_to_messages[node]

    def get_memory(self, node_idxes: Union[List[int], np.ndarray]) -> torch.Tensor:
        return self.memory[node_idxes, :]

    def set_memory(self, node_idxes: Union[List[int], np.ndarray], values: torch.Tensor) -> None:
        self.memory[node_idxes, :] = values

    def get_last_update(self, node_idxes: List[int]) -> torch.Tensor:
        return self.last_update[node_idxes]

    def backup_memory(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[Message]]]:
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup: Tuple[torch.Tensor, torch.Tensor, Dict[int, List[Message]]]) -> None:
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self) -> None:
        self.memory.detach_()

        # Detach all stored messages
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))

            self.messages[k] = new_node_messages

    def clear_messages(self, nodes: np.ndarray) -> None:
        for node in nodes:
            self.messages[node] = []
