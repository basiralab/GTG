import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class VGAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGAE, self).__init__()
        # Encoder
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def forward(self, x, edge_index):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        adj_recon = self.decode(z)
        return adj_recon, mu, logstd


class VGAEDataset(torch.utils.data.Dataset):
    def __init__(self, source_data, target_data):
        self.source_data = source_data
        self.target_data = target_data

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        src = self.source_data[idx]
        trg = self.target_data[idx]
        if isinstance(src, dict):
            src_pyg = src['pyg']
            target_adj = trg['mat']
        else:
            src_pyg = src
            target_adj = trg.adj
        x = src_pyg.x
        edge_index = src_pyg.edge_index
        return x, edge_index, target_adj
