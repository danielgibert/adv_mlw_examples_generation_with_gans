import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorNetwork(nn.Module):

    def __init__(self, discriminator_parameters:dict):
        super(DiscriminatorNetwork, self).__init__()
        self.discriminator_parameters = discriminator_parameters
        self.features_size = 256

        self.dropout_input = torch.nn.Dropout(p=self.discriminator_parameters["input_dropout_rate"])


        self.dense1 = nn.Linear(
            in_features=self.features_size,
            out_features=self.discriminator_parameters["hidden_neurons"][0]
        )
        
        self.dropout1 = torch.nn.Dropout(p=self.discriminator_parameters["dropout_rate"])

        self.dense2 = nn.Linear(
            in_features=self.discriminator_parameters["hidden_neurons"][0],
            out_features=self.discriminator_parameters["hidden_neurons"][1]
        )
        self.dropout2 = torch.nn.Dropout(p=self.discriminator_parameters["dropout_rate"])

        self.dense3 = nn.Linear(
            in_features=self.discriminator_parameters["hidden_neurons"][1],
            out_features=1
        )
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.dropout_input(x)
        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.sigmoid(x)
        return x