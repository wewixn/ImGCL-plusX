import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, degree, subgraph, to_dense_adj

from imblearn.under_sampling import ClusterCentroids, TomekLinks
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from Contrast import WithinEmbedContrast
# from GCL.models.contrast_model import WithinEmbedContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import DBLP, Amazon
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GConv, self).__init__()
        self.act = torch.nn.PReLU()
        self.bn = torch.nn.BatchNorm1d(2 * hidden_dim, momentum=0.01)
        self.conv1 = GCNConv(input_dim, 2 * hidden_dim, cached=False)
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.conv1(x, edge_index, edge_weight)
        z = self.bn(z)
        z = self.act(z)
        z = self.conv2(z, edge_index, edge_weight)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2


def remap_edge_index(edge_index, mask, prev_dim):
    device = edge_index.device
    mask = torch.unique(mask)
    mapping = torch.zeros(prev_dim, dtype=torch.long).to(device=device)
    mapping[mask] = torch.arange(mask.size(0)).to(device=device)
    remapped_edge_index = mapping[edge_index]

    return remapped_edge_index


def pbs_sample(data, pseudo_labels, a, alpha=0.85, pr=0.1, l=0.1):
    device = data.edge_index.device
    counts = torch.bincount(pseudo_labels)
    p_pb = a * counts / data.num_nodes + (1-a) / counts.size(0)

    edge_self_loop, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    row, col = edge_self_loop
    deg = degree(col, data.num_nodes, dtype=data.x.dtype)
    deg_inv = deg.pow(-1)
    adj_matrix = to_dense_adj(edge_self_loop, edge_attr=data.edge_attr)
    node_centrality = alpha * adj_matrix @ deg_inv + torch.ones(data.num_nodes).to(device)
    nc_max = torch.max(node_centrality)
    nc_min = torch.min(node_centrality)
    p_pbs = (nc_max - node_centrality) / (nc_max - nc_min) * p_pb[pseudo_labels]
    pr = torch.tensor(pr, dtype=torch.float32)
    p_pbs = torch.maximum(p_pbs, pr)

    samples = torch.multinomial(p_pbs, int(data.num_nodes*l))[0]

    return samples


def cluster(encoder_model, data, num_clusters=100):
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)

    data_array = z.cpu().detach().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data_array)
    cluster_labels = torch.from_numpy(cluster_labels)

    return cluster_labels


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets', 'Amazon')
    dataset = Amazon(path, name='Photo', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    data_train = data.clone()

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=256).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=5e-4)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=400,
        max_epochs=4000)

    B = 100
    with tqdm(total=4000, desc='(T)') as pbar:
        for epoch in range(1, 4001):
            loss = train(encoder_model, contrast_model, data_train, optimizer)
            if epoch % B == 0:
                num_clusters = min(int(data.x.size(0) / 100), 100)
                pseudo_labels = cluster(encoder_model, data, num_clusters=num_clusters).to(device)
                sample_mask = pbs_sample(data, pseudo_labels, 1 - epoch / 40)
                data_train.x = data.x[sample_mask]
                data_train.edge_index = subgraph(sample_mask, data.edge_index, num_nodes=data.x.size(0))[0]
                data_train.edge_index = remap_edge_index(data_train.edge_index, sample_mask, data.x.size(0))

            scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    return test_result


if __name__ == '__main__':
    main()
