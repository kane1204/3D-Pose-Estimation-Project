from torch import nn
# Model For implementing simple-effective-3D pose-baseline
class LiftingNet(nn.Module):
    # Based of https://arxiv.org/abs/1705.03098
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
        print(x)
        x = self.flatten(x)
        residual = x
        out = self.seq1(x)
        out+=residual
        out = self.seq2(out)
        return self.out(x)


# Taken from https://github.com/wuyenlin/SimpleBaseline/blob/main/common/model.py
class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,input, out,
                 linear_size=992,
                 num_stage=2,
                 p_dropout=0.5):
        super(LinearModel, self).__init__()
        
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size = input
        # 3d joints
        self.output_size = out

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y