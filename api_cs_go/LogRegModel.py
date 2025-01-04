import torch
import torch.nn as nn
from dataframes import team_data
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(LogisticRegressionModel, self).__init__()
        layers = []


        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())


        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x))


def evaluate_model(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    return accuracy


checkpoint = torch.load("model_no_id_705.pth")
model = LogisticRegressionModel(input_dim=team_data.shape[1]-1, hidden_dims=[32, 16])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
