from abc import abstractmethod
from typing import Any

import torch
from torch import nn


class MessageFunction(nn.Module):
    """
    Module which computes the message for a given interaction.
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    @abstractmethod
    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLPMessageFunction(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int) -> None:
        super(MLPMessageFunction, self).__init__()

        self.mlp = self.layers = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
        )

    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        messages = self.mlp(raw_messages)
        return messages


class IdentityMessageFunction(MessageFunction):
    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        return raw_messages


def get_message_function(module_type: str, raw_message_dimension: int, message_dimension: int) -> MessageFunction:
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    elif module_type == "identity":
        return IdentityMessageFunction()
