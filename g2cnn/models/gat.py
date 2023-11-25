import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

class GATNet(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super(GATNet, self).__init__()



        # Graph Attention Network (GAT) for drugs
        self.gcn1 = GATConv(78, 144, heads=10, dropout=dropout)
        self.gcn2 = GATConv(144 * 10, 144, dropout=dropout)
        self.fc_g1 = nn.Linear(144, 144)

        self.size = 8
        self.Protein_CNNs = nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=self.size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2))

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(288, 144)
        self.fc2 = nn.Linear(144, 72)
        self.fc3 = nn.Linear(72, 1)
        # self.out = nn.Linear(36, 1)

    def forward(self, data):
        # Graph input (drugs) feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        # print(x.shape)

        targets = data.target
        targets = targets.float()
        # print(targets.shape)
        self.size = targets.shape[0]
        protcnn = self.Protein_CNNs(targets)
        protcnn = protcnn.view(protcnn.shape[0], protcnn.shape[1] * protcnn.shape[2])
        pair = torch.cat([x, protcnn], dim=1)
        
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        predict = self.fc3(fully2)
        # predict = self.out(fully3)
        
        return predict