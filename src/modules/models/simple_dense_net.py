from torch import nn

class SimpleDenseNet(nn.Module):
    def __init__(self, hparams : dict):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["hidden_size"]),
            nn.ReLU(),
            nn.Linear(hparams["hidden_size"], hparams["output_size"])
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))