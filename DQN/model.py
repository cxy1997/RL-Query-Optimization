import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1: 4])
        fan_out = np.prod(weight_shape[2: 4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class TableFeatureExtractor(nn.Module):
    def __init__(self, n_tables=21, hidden_size=64):
        super(TableFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(n_tables, hidden_size-1)
        # self.ln1 = nn.LayerNorm([hidden_size-1])
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size-1)
        # self.ln2 = nn.LayerNorm([hidden_size-1])
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.apply(weights_init)

    def forward(self, one_hot_table, cardinality):
        c = torch.tensor(np.log(cardinality+1)-9).view(1, 1).to(one_hot_table.device).float()

        x = self.fc1(one_hot_table.float())
        # x = self.ln1(x)
        x = self.relu1(x)
        x = torch.cat([x, c], dim=1)

        x = self.fc2(x)
        # x = self.ln2(x)
        x = self.relu2(x)
        x = torch.cat([x, c], dim=1)

        x = self.fc3(x)
        return x



class Net(nn.Module):
    def __init__(self, n_tables=21, n_columns=35, hidden_size=64):
        super(Net, self).__init__()
        self.n_tables = n_tables
        self.n_columns = n_columns
        self.hidden_size = hidden_size

        self.table_feature_extractor = TableFeatureExtractor(n_tables, hidden_size)
        self.column_feature_extractor = nn.Linear(n_columns, hidden_size, bias=False)
        self.aggregate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            # nn.LayerNorm([hidden_size]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm([hidden_size]),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            # nn.LayerNorm([hidden_size]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm([hidden_size]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1, bias=False),
        )
        # self.apply(weights_init)

    def forward(self, state):
        actions = state["possible_actions"]

        table_features = []
        for t, c in state["tables"]:
            t = torch.from_numpy(t).view(1, self.n_tables).float()
            if torch.cuda.is_available():
                t = t.cuda()
            table_features.append(self.table_feature_extractor(t, c))

        Q_dict = dict()
        for k, v in actions.items():
            c1 = torch.from_numpy(v[0]).view(1, self.n_columns).float()
            c2 = torch.from_numpy(v[1]).view(1, self.n_columns).float()
            if torch.cuda.is_available():
                c1 = c1.cuda()
                c2 = c2.cuda()
            c1 = self.column_feature_extractor(c1)
            c2 = self.column_feature_extractor(c2)
            c1 = torch.cat([table_features[k[0]], c1], dim=1)
            c2 = torch.cat([table_features[k[1]], c2], dim=1)
            c1 = self.aggregate_layer(c1)
            c2 = self.aggregate_layer(c2)
            Q_dict[k] = self.final_layer(torch.cat([c1, c2], dim=1))
        return Q_dict


if __name__ == "__main__":
    f1 = TableFeatureExtractor().cuda()
    print(f1.device)
    t = torch.ones(1, 21).cuda()
    c = 100
    y = f1(t, 1)
    print(y.shape)
