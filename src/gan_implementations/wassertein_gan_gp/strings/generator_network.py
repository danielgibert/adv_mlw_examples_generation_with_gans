import torch
import torch.nn as nn

class GeneratorNetwork(nn.Module):

    def __init__(self, generator_parameters:dict):
        super(GeneratorNetwork, self).__init__()
        self.generator_parameters = generator_parameters
        self.features_size = generator_parameters["features_size"]

        self.dropout_input = torch.nn.Dropout(p=self.generator_parameters["input_dropout_rate"])
        self.dense1 = nn.Linear(
            in_features=self.features_size+self.generator_parameters["z_size"],
            out_features=self.generator_parameters["hidden_neurons"][0]
        )
        self.dropout1 = torch.nn.Dropout(p=self.generator_parameters["dropout_rate"])
        self.dense2 = nn.Linear(
            in_features=self.generator_parameters["hidden_neurons"][0],
            out_features=self.generator_parameters["hidden_neurons"][1]
        )
        self.dropout2 = torch.nn.Dropout(p=self.generator_parameters["dropout_rate"])
        self.dense3 = nn.Linear(
            in_features=self.generator_parameters["hidden_neurons"][1],
            out_features=self.features_size
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        #print("Concatenated vector: {}".format(x.shape))
        x = torch.cat(input, dim=1)
        x = self.dropout_input(x)
        x = self.dense1(x)
        x = self.relu(x)
        #print("Dense 1: {}".format(x.shape))

        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu(x)
        #print("Dense 2: {}".format(x.shape))

        x = self.dropout2(x)
        x = self.dense3(x)

        x = self.sigmoid(x)
        #print("Sigmoid: {}".format(x.shape))
        x = torch.max(x, input[0])

        return x
