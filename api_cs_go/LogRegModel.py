import torch
import torch.nn as nn
from dataframes import team_data
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def evaluate_model(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    return accuracy


checkpoint = torch.load("model_id_combined_73.pth")
model = LogisticRegressionModel(input_dim=team_data.shape[1])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
