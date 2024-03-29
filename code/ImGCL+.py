import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, degree, subgraph, to_dense_adj

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from Contrast import WithinEmbedContrast
# from GCL.models.contrast_model import WithinEmbedContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import DBLP, Amazon, WikiCS
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


def sim_sample(data, pseudo_labels, sampler=None, encoder_model=None):
    data_clone = data.clone()

    if encoder_model is not None:
        data4sample, _, _ = encoder_model(data_clone.x, data_clone.edge_index, data_clone.edge_attr)
        data4sample = data4sample.detach().cpu().numpy()
    else:
        data4sample = data_clone.x.detach().cpu().numpy()

    if sampler is None:
        sampler = TomekLinks()

    _, pseudo_labels = sampler.fit_resample(data4sample, pseudo_labels.cpu().numpy())
    pseudo_labels = torch.from_numpy(pseudo_labels).to(device=data.x.device)
    sample_mask = sampler.sample_indices_

    sample_mask = torch.from_numpy(sample_mask).to(device=data.x.device)
    data.x = data_clone.x[sample_mask]
    data.edge_index = subgraph(sample_mask, data_clone.edge_index, num_nodes=data_clone.x.size(0))[0]
    data.edge_index = remap_edge_index(data.edge_index, sample_mask, data_clone.x.size(0))

    return data, pseudo_labels


def cluster(encoder_model, data, num_clusters=100):
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)

    data_array = z.cpu().detach().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data_array)
    cluster_labels = torch.from_numpy(cluster_labels)

    return cluster_labels


def node_perturbation(nodes_feature, noise_std=0.1):
    nodes_feature += noise_std * torch.randn_like(nodes_feature) / nodes_feature.size(1)
    return nodes_feature


def edge_perturbation(edge_index, mask, disconnect_prob=0.05):
    selected_edges = edge_index[:, mask]
    disconnect_mask = np.random.rand(selected_edges.shape[1]) > disconnect_prob
    disconnected_edges = selected_edges[:, disconnect_mask]
    unchanged_edges = edge_index[:, ~mask]
    edge_index = np.concatenate([unchanged_edges, disconnected_edges], axis=1)
    return edge_index


def pos_perturbation(data, ):
    data = T.RandomRotate(degrees=10, axis=0)(data)
    data = T.RandomTranslate(0.1)(data)
    return data


def rand_noise(data, labels):
    labels = labels.cpu().numpy()
    cluster_counts = np.bincount(labels)
    cluster_threshold = 1 / len(cluster_counts)
    majority_clusters = np.where((cluster_counts / len(labels)) > cluster_threshold)[0]

    majority_indices = np.concatenate([np.where(labels == cluster)[0] for cluster in majority_clusters])
    majority_indices = majority_indices.tolist()

    # add noise to data.x
    data.x[majority_indices] = node_perturbation(data.x[majority_indices])

    # add noise to data.edge_index
    data.edge_index = data.edge_index.cpu().numpy()
    mask = np.isin(data.edge_index[0], majority_indices) | np.isin(data.edge_index[1], majority_indices)
    data.edge_index = edge_perturbation(data.edge_index, mask, disconnect_prob=0.01)
    data.edge_index = torch.from_numpy(data.edge_index).to(device=data.x.device)

    # add noise to position
    # data = pos_perturbation(data, )

    return data


def generate_sampling_strategy(labels):
    labels = labels.cpu().numpy()
    class_counts = np.bincount(labels)
    class_threshold = len(labels) / len(class_counts)
    max_minority_count = min(class_counts)
    for class_count in enumerate(class_counts):
        if class_count < class_threshold:
            max_minority_count = max(max_minority_count, class_count)
    sampling_strategy = {}

    for class_label, class_count in enumerate(class_counts):
        if class_count < class_threshold:
            sampling_strategy[class_label] = class_count
        else:
            sampling_strategy[class_label] = max(int(0.5 * class_count), max_minority_count)

    return sampling_strategy


def train(encoder_model, contrast_model, data, optimizer, pseudo_labels=None):
    encoder_model.train()
    optimizer.zero_grad()
    _, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(z1, z2, pseudo_labels)
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
    dataset = Amazon(path, name='Computers', transform=T.NormalizeFeatures())
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
    num_clusters = min(int(data.x.size(0) / 100), 100)
    pseudo_labels = cluster(encoder_model, data, num_clusters=num_clusters).to(device)
    with tqdm(total=4000, desc='(T)') as pbar:
        for epoch in range(1, 4001):
            loss = train(encoder_model, contrast_model, data_train, optimizer)
            if epoch % B == 0:
                # num_clusters = min(int(data.x.size(0) / 100), 100)
                # pseudo_labels = cluster(encoder_model, data, num_clusters=num_clusters).to(device)
                # sample_mask = pbs_sample(data, pseudo_labels, 1-epoch/40)
                # data_train.x = data.x[sample_mask]
                # data_train.edge_index = subgraph(sample_mask, data.edge_index, num_nodes=data.x.size(0))[0]
                # data_train.edge_index = remap_edge_index(data_train.edge_index, sample_mask, data.x.size(0))
                # pseudo_labels = pseudo_labels[sample_mask]

                prev_data_count = data_train.x.size(0)
                undersampler = TomekLinks()
                data_train, pseudo_labels = sim_sample(data_train, pseudo_labels, undersampler, encoder_model)

                # data_train = rand_noise(data_train, pseudo_labels)

                if data_train.x.size(0) >= 0.99 * prev_data_count:
                    data_train = data.clone()
                    pseudo_labels = cluster(encoder_model, data, num_clusters=num_clusters).to(device)

            scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    return test_result


if __name__ == '__main__':
    res = []
    for i in range(4):
        res.append(main())

    print(res)
