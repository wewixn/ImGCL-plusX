import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.cluster import KMeans
from torch_geometric.utils import add_self_loops, degree, subgraph, to_dense_adj, dense_to_sparse

import random
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from imblearn.under_sampling import TomekLinks, NeighbourhoodCleaningRule, NearMiss
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models.contrast_model import WithinEmbedContrast
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Amazon, WikiCS, Planetoid
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_hidden):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, num_hidden, heads=8, concat=True)
        self.conv2 = GATConv(num_hidden * 8, num_hidden, heads=1, concat=False)

    def forward(self, x, edge_index, edge_weight=None):
        z = F.dropout(x, p=0.6, training=self.training)
        z = self.conv1(z, edge_index, edge_weight)
        z = F.elu(z)
        z = F.dropout(z, p=0.6, training=self.training)
        z = self.conv2(z, edge_index, edge_weight)
        return z


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
    adj_matrix = to_dense_adj(edge_self_loop, edge_attr=data.edge_attr)[0]
    node_centrality = alpha * adj_matrix @ deg_inv + torch.ones(data.num_nodes).to(device)
    nc_max = torch.max(node_centrality)
    nc_min = torch.min(node_centrality)
    p_pbs = (nc_max - node_centrality) / (nc_max - nc_min) * p_pb[pseudo_labels]
    pr = torch.tensor(pr, dtype=torch.float32)
    p_pbs = torch.maximum(p_pbs, pr)

    samples = torch.multinomial(p_pbs, int(data.num_nodes*l))[0]

    return samples


def sim_sample(data, pseudo_labels, sampler=None, encoder_model=None):
    device = data.x.device
    data_clone = data.clone()

    if encoder_model is not None:
        data4sample = encoder_model(data_clone.x, data_clone.edge_index, data_clone.edge_attr)
        data4sample = data4sample.detach().cpu().numpy()
    else:
        data4sample = data_clone.x.detach().cpu().numpy()

    if sampler is None:
        sampler = TomekLinks()

    _, pseudo_labels = sampler.fit_resample(data4sample, pseudo_labels.cpu().numpy())
    pseudo_labels = torch.from_numpy(pseudo_labels).to(device=device)
    sample_mask = torch.from_numpy(sampler.sample_indices_).to(device=device)

    data_clone.x = data.x[sample_mask]
    data_clone.edge_index = subgraph(sample_mask, data.edge_index, num_nodes=data.num_nodes)[0]
    data_clone.edge_index = remap_edge_index(data_clone.edge_index, sample_mask, data.num_nodes)

    return data_clone, pseudo_labels


def src_smote(adj, features, labels, portion=1.0, im_class_num=3):
    cluster_counts = torch.bincount(labels)
    avg_number = int(labels.size(0) / cluster_counts.size(0))
    minority_clusters = cluster_counts[cluster_counts < avg_number].size(0)
    im_class_num = min(im_class_num, minority_clusters)

    sample_idx = torch.argsort(cluster_counts)[:im_class_num]
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    idx_train = torch.arange(labels.shape[0]).to(device=adj.device)

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (sample_idx[i]))[idx_train]]
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])

            portion_rest = (avg_number / new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion

        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

        num = int(new_chosen.shape[0] * portion_rest)
        new_chosen = new_chosen[:num]
        if num == 0:
            continue

        chosen_embed = features[new_chosen, :]
        distance = squareform(pdist(chosen_embed.cpu().detach()))
        np.fill_diagonal(distance, distance.max() + 100)

        idx_neighbor = distance.argmin(axis=-1)

        interp_place = random.random()
        embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed), 0)

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:, :]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]

    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    adj = new_adj

    return adj, features, labels


def over_sample(data, pseudo_labels, portion=0.0):
    device = data.x.device
    data_clone = data.clone()

    # align dims
    adj_part = to_dense_adj(data.edge_index)[0]
    adj = torch.zeros((data_clone.num_nodes, data_clone.num_nodes)).to(device=device)
    adj[:adj_part.size(0), :adj_part.size(1)] = adj_part
    adj, features, labels = src_smote(adj, data_clone.x, pseudo_labels, portion=portion)
    data_clone.x = features
    data_clone.edge_index = dense_to_sparse(adj)[0]

    return data_clone, pseudo_labels


def cluster(encoder_model, data, num_clusters=10):
    device = data.x.device
    z = encoder_model(data.x, data.edge_index, data.edge_attr)

    data_array = z.cpu().detach().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data_array)

    # process small clusters
    min_cluster_size = min(30, data.x.size(0) / 3 / num_clusters)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    small_clusters = unique_labels[counts < min_cluster_size].tolist()
    centers = kmeans.cluster_centers_
    while small_clusters:
        small_cluster = small_clusters[0]
        distances = np.linalg.norm(centers - centers[small_cluster], axis=1)
        distances[small_cluster] = np.inf
        nearest_cluster = np.argmin(distances)
        centers[nearest_cluster] = ((counts[nearest_cluster] * centers[nearest_cluster] +
                                     counts[small_cluster] * centers[small_cluster]) /
                                    (counts[nearest_cluster] + counts[small_cluster]))
        counts[nearest_cluster] += counts[small_cluster]
        centers = np.delete(centers, small_cluster, axis=0)
        counts = np.delete(counts, small_cluster)
        cluster_labels[cluster_labels == small_cluster] = nearest_cluster
        cluster_labels[cluster_labels > small_cluster] -= 1
        unique_labels = np.unique(cluster_labels)
        small_clusters = unique_labels[counts < min_cluster_size].tolist()

    cluster_labels = torch.from_numpy(cluster_labels).to(device)
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

    return data


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
    path = osp.join(osp.expanduser('.'), 'datasets', 'Amazon')
    dataset = Amazon(path, name='computers', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=256).to(device)
    gat = GAT(num_features=dataset.num_features, num_hidden=64).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=5e-4)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=400,
        max_epochs=4000)

    B = 200
    data_train = data.clone()
    with tqdm(total=4000, desc='(T)') as pbar:
        for epoch in range(1, 4001):
            loss = train(encoder_model, contrast_model, data_train, optimizer)
            optimizer.zero_grad()
            if epoch % B == 0:
                # num_clusters = min(int(data.x.size(0) / 100), 100)
                # pseudo_labels = cluster(encoder_model.encoder, data, num_clusters=num_clusters).to(device)
                # sample_mask = pbs_sample(data, pseudo_labels, 1-epoch/40)
                # data_train.x = data.x[sample_mask]
                # data_train.edge_index = subgraph(sample_mask, data.edge_index, num_nodes=data.x.size(0))[0]
                # data_train.edge_index = remap_edge_index(data_train.edge_index, sample_mask, data.x.size(0))
                # pseudo_labels = pseudo_labels[sample_mask]

                if data_train.x.size(0) <= 0.2 * data.x.size(0):
                    data_train = data.clone()

                num_clusters = 100
                pseudo_labels = cluster(encoder_model.encoder, data_train, num_clusters=num_clusters)
                if torch.unique(pseudo_labels).size(0) == 1:
                    data_train = data.clone()
                    pseudo_labels = cluster(encoder_model.encoder, data_train, num_clusters=num_clusters)
                undersampler = NearMiss(version=3, sampling_strategy='majority', n_neighbors_ver3=20)
                data_train, pseudo_labels = sim_sample(data_train, pseudo_labels, undersampler, encoder_model.encoder)
                data_train, pseudo_labels = over_sample(data_train, pseudo_labels, portion=0.5)

                save_path = f'data/saved_data_epoch_{epoch}.pt'
                torch.save(data_train, save_path)

            scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    return test_result


if __name__ == '__main__':
    res = []
    for i in range(5):
        res.append(main())

    for i in range(5):
        print(f'F1Mi={res[i]["micro_f1"]:.4f}, F1Ma={res[i]["macro_f1"]:.4f}')
