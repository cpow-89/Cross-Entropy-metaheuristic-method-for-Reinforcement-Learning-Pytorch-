import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod


class BaseNetwork(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.config = config
        torch.manual_seed(config["seed"])

    def set_weights(self, weights):
        """override current weights with given weight list values"""
        weights = iter(weights)
        for param in self.network.parameters():
            param.data.copy_(torch.from_numpy(next(weights)))

    def get_weights(self):
        """get current weights as list"""
        return [param.detach().numpy() for param in self.network.parameters()]

    @abstractmethod
    def forward(self, x):
        pass


class ContinuousPolicyNetwork(BaseNetwork):
    """Feedforward multilayer perceptron"""
    def __init__(self, config):
        super().__init__(config)
        self.network = nn.Sequential(nn.Linear(config["observation_space"], config["hidden_nodes"]),
                                     getattr(nn, config["fc1_activation_func"])(),
                                     nn.Linear(config["hidden_nodes"], config["action_space"]),
                                     getattr(nn, config["output_activation_func"])())

        self.network.eval()

    def forward(self, x):
        """run network forward path"""
        x = self.network(x)
        return x.cpu().data


class DiscretePolicyNetwork(BaseNetwork):
    """Feedforward multilayer perceptron"""
    def __init__(self, config):
        super().__init__(config)
        self.network = nn.Sequential(nn.Linear(config["observation_space"], config["hidden_nodes"]),
                                     getattr(nn, config["fc1_activation_func"])(),
                                     nn.Linear(config["hidden_nodes"], config["action_space"]))

        self.network.eval()

    def forward(self, x):
        """run network forward path"""
        x = self.network(x)
        x = F.softmax(x, dim=0)
        action = torch.argmax(x, dim=0)
        return action.item()
