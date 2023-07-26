import torch.nn as nn
import abc

class InceptionNetwork(nn.Module):
    def __init__(self, network_parameters:dict):
        super(InceptionNetwork, self).__init__()
        self.network_parameters = network_parameters

    @abc.abstractmethod
    def retrieve_activations(self, x):
        """
        Retrieve the intermediate features of the Inception network substitutor
        :param x:
        :return:
        """
        pass

