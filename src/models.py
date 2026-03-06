import torch
import torch.nn as nn


# Define the FCBlock class with Kaiming initialization
class FCBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features=in_feats, out_features=out_feats)
        self.relu = nn.ReLU()

        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.fc.weight, mode="fan_in", nonlinearity="relu")

        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.relu(self.fc(x))


class FedXGBllrCNN(nn.Module):
    def __init__(
            self,
            num_clients: int,
            trees_per_client: int,
            in_channels: int = 1,
            conv_channels: int = 64,
            fc_layers: list = [],
            out_channels: int = 1,
            dropout_rate: float = 0.5
        ):
        """
        Initializes the one-layer 1D CNN for FedXGBllr.

        Args:
            num_clients (int): K, the total number of clients participating.
            trees_per_client (int): M, the number of trees in each client's ensemble.
                                    (Note: The source's general setup uses 500 trees total divided by clients,
                                    but your query specifies 1000 estimators *each*.) [6]
            num_conv_channels (int): The number of output channels for the 1D convolution layer. [3]
        """
        super(FedXGBllrCNN, self).__init__()

        # Calculate the total number of trees across all aggregated ensembles [7]
        self.total_trees = num_clients * trees_per_client

        # The kernel size and stride of the 1D convolution are equal to
        # the number of trees (M) in each client's tree ensemble. [2]
        self.kernel_size = trees_per_client
        self.stride = trees_per_client

        # The 1D convolution layer with 1 input channel (for prediction outcomes) [2]
        # and a specified number of output channels.
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,        # Input: prediction outcomes of all trees (treated as a sequence) [2]
            out_channels=conv_channels,     # Number of learning rate strategies [3, 4]
            kernel_size=self.kernel_size,   # Equal to M (trees_per_client) [2]
            stride=self.stride,             # Equal to M (trees_per_client) [2]
            padding=0
        )
        
        # Activation function G, set to ReLU to avoid overfitting. [5]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Adjust dropout rate as needed

        # Calculate the flattened dimension for the Fully Connected (FC) layer [4]
        conv_output_length = (self.total_trees - self.kernel_size) // self.stride + 1
        flattened_dimension = conv_channels * conv_output_length

        # Output layers
        fc_layers, fc_blocks = [flattened_dimension, *fc_layers, out_channels], []
        for in_feats, out_feats in zip(fc_layers[:-2], fc_layers[1:-1]):
            fc_blocks.append(FCBlock(in_feats, out_feats))
        
        # Add the final linear layer without ReLU activation
        fc_blocks.append(nn.Linear(in_features=fc_layers[-2], out_features=fc_layers[-1]))
        self.fc = nn.Sequential(*fc_blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 1D CNN.

        Args:
            x (torch.Tensor): Input tensor of prediction outcomes from all trees.
                              Expected shape: (batch_size, 1, total_trees)

        Returns:
            torch.Tensor: The final predicted output.
        """
        # Apply 1D convolution
        x = self.conv1d(x)  # Shape: (batch_size, num_conv_channels, conv_output_length)
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, start_dim=1) # Shape: (batch_size, flattened_dimension)
        
        # Apply ReLU activation
        x = self.relu(x)    # Shape: (batch_size, num_conv_channels, conv_output_length)
        
        # Apply Dropout
        x = self.dropout(x)

        # Apply the fully connected layer to get the final prediction
        x = self.fc(x)      # Shape: (batch_size, 1)

        return x
