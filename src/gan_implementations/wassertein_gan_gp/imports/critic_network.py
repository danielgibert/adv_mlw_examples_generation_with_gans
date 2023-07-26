import torch
import torch.nn as nn

class CriticNetwork(nn.Module):

    def __init__(self, critic_parameters:dict):
        super(CriticNetwork, self).__init__()
        self.critic_parameters = critic_parameters
        self.features_size = critic_parameters["features_size"]

        self.dropout_input = torch.nn.Dropout(p=self.critic_parameters["input_dropout_rate"])


        self.dense1 = nn.Linear(
            in_features=self.features_size,
            out_features=self.critic_parameters["hidden_neurons"][0]
        )
        
        self.dropout1 = torch.nn.Dropout(p=self.critic_parameters["dropout_rate"])

        self.dense2 = nn.Linear(
            in_features=self.critic_parameters["hidden_neurons"][0],
            out_features=self.critic_parameters["hidden_neurons"][1]
        )
        self.dropout2 = torch.nn.Dropout(p=self.critic_parameters["dropout_rate"])

        self.dense3 = nn.Linear(
            in_features=self.critic_parameters["hidden_neurons"][1],
            out_features=self.critic_parameters["hidden_neurons"][2]
        )
        self.dropout3 = torch.nn.Dropout(p=self.critic_parameters["dropout_rate"])

        self.dense4 = nn.Linear(
            in_features=self.critic_parameters["hidden_neurons"][2],
            out_features=1
        )
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.dropout_input(x)
        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.leaky_relu(x)
        x = self.dropout3(x)
        x = self.dense4(x)
        return x