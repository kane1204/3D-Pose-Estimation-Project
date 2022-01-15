from torch import nn
class LiftingNet(nn.Module):
    def __init__(self,n_inputs,hidden,n_output):
        super(LiftingNet, self).__init__()
        self.flatten = nn.Flatten()
        self.seq1 = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, n_inputs)
        )
        self.seq2 = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, n_inputs)
        )
        self.out = nn.Linear(n_inputs, n_output)
    def forward(self, x):
        x = self.flatten(x)
        residual = x
        out = self.seq1(x)
        out+=residual
        out = self.seq2(out)
        return self.out(x)