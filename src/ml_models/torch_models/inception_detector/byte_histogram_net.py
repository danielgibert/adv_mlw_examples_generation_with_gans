from src.ml_models.torch_models.inception_detector.inception_net import InceptionNetwork
import torch
import torch.nn as nn


class ByteHistogramNetwork(InceptionNetwork):
    def __init__(self, network_parameters:dict):
        super(ByteHistogramNetwork, self).__init__(network_parameters)

        self.dropout_input = torch.nn.Dropout(p=self.network_parameters["input_dropout_rate"])

        self.dense1 = nn.Linear(
            in_features=self.network_parameters["input_features"],
            out_features=self.network_parameters["hidden_neurons"][0]
        )

        self.dropout1 = torch.nn.Dropout(p=self.network_parameters["dropout_rate"])

        self.dense2 = nn.Linear(
            in_features=self.network_parameters["hidden_neurons"][0],
            out_features=self.network_parameters["hidden_neurons"][1]
        )

        self.dropout2 = torch.nn.Dropout(p=self.network_parameters["dropout_rate"])

        self.dense3 = nn.Linear(
            in_features=self.network_parameters["hidden_neurons"][1],
            out_features=self.network_parameters["output_classes"]
        )
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout_input(x)
        x = self.dense1(x)
        x = self.elu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.elu(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.sigmoid(x)
        return x

    def retrieve_features(self, x):
        x = self.dropout_input(x)
        x = self.dense1(x)
        x = self.elu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        return x