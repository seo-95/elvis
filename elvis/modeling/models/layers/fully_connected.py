import torch.nn as nn

__all__ = ['FC',
           'MLP']


class FC(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0., use_relu=True):
        super(FC, self).__init__()
        
        self.fc = [nn.Linear(in_features=in_features, out_features=out_features)]
        if use_relu:
            self.fc.append(nn.ReLU(inplace=True))
        if dropout_p > 0:
            self.fc.append(nn.Dropout(p=dropout_p))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, dropout_p=0., use_relu=True):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(FC(in_features=in_features, 
                                    out_features=hidden_dim,
                                    dropout_p=dropout_p,
                                    use_relu=use_relu),
                                nn.Linear(in_features=hidden_dim, 
                                        out_features=out_features))

    def forward(self, x):
        return self.mlp(x)