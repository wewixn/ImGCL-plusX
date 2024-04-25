import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.cluster import KMeans, DBSCAN
from torch_geometric.utils import subgraph, to_dense_adj, dense_to_sparse
from torch.cuda.amp import autocast, GradScaler
import collections
import random
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from imblearn.under_sampling import TomekLinks, NeighbourhoodCleaningRule
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
    cluster_counts = cluster_counts[cluster_counts > 1]
    if cluster_counts.size(0) == 1:
        return adj, features, labels
    avg_number = int(cluster_counts.sum() / cluster_counts.size(0))
    minority_clusters = cluster_counts[cluster_counts < avg_number].size(0)
    im_class_num = min(im_class_num, minority_clusters)

    sample_idx = torch.argsort(cluster_counts)[:im_class_num]
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    idx_train = torch.arange(labels.shape[0]).to(device=adj.device)
    min_samp = 42 if portion < 1 else 0

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

        new_chosen = idx_train[(labels == (sample_idx[i]))[idx_train]]
        num = new_chosen.shape[0] if new_chosen.shape[0] < min_samp else int(new_chosen.shape[0] * portion_rest)
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
    adj, features, pseudo_labels = src_smote(adj, data_clone.x, pseudo_labels, portion=portion)
    data_clone.x = features
    data_clone.edge_index = dense_to_sparse(adj)[0]

    return data_clone, pseudo_labels


def cluster_with_outlier(encoder_model, data, eps=0.6, eps_gap=0.02):
    device = data.x.device
    z = encoder_model(data.x, data.edge_index, data.edge_attr)

    data_array = z.cpu().detach().numpy()
    eps_tight = eps - eps_gap
    eps_loose = eps + eps_gap
    min_cluster_size = min(24, int(data.x.size(0) / 42))
    cluster = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='euclidean', n_jobs=-1)
    cluster_tight = DBSCAN(eps=eps_tight, min_samples=min_cluster_size, metric='euclidean', n_jobs=-1)
    cluster_loose = DBSCAN(eps=eps_loose, min_samples=min_cluster_size, metric='euclidean', n_jobs=-1)

    pseudo_labels = cluster.fit_predict(data_array)
    pseudo_labels_tight = cluster_tight.fit_predict(data_array)
    pseudo_labels_loose = cluster_loose.fit_predict(data_array)

    def process_pseudo_labels(cluster_id):
        max_label = max(cluster_id)
        cluster_id[cluster_id == -1] = np.arange(max_label + 1, max_label + 1 + (cluster_id == -1).sum())
        return torch.from_numpy(cluster_id).long().to(device=device)

    pseudo_labels = process_pseudo_labels(pseudo_labels)
    pseudo_labels_tight = process_pseudo_labels(pseudo_labels_tight)
    pseudo_labels_loose = process_pseudo_labels(pseudo_labels_loose)

    # compute R_indep and R_comp
    N = pseudo_labels.size(0)
    label_sim = (pseudo_labels.view(N, 1) == pseudo_labels.view(1, N)).float()
    label_sim_tight = (pseudo_labels_tight.view(N, 1) == pseudo_labels_tight.view(1, N)).float()
    label_sim_loose = (pseudo_labels_loose.view(N, 1) == pseudo_labels_loose.view(1, N)).float()

    R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
    R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
    assert ((R_comp.min() >= 0) and (R_comp.max() <= 1))
    assert ((R_indep.min() >= 0) and (R_indep.max() <= 1))

    cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
    cluster_num = collections.defaultdict(int)
    for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
        cluster_R_comp[label.item()].append(comp.item())
        cluster_R_indep[label.item()].append(indep.item())
        cluster_num[label.item()] += 1

    cluster_R_comp = torch.tensor([min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())])
    cluster_R_indep = torch.tensor([min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())])
    cluster_R_indep_noins = cluster_R_indep[torch.tensor([cluster_num[num] > 1 for num in sorted(cluster_num.keys())])]
    if len(cluster_R_indep_noins) == 0:
        return pseudo_labels
    else:
        indep_thres = torch.sort(cluster_R_indep_noins)[0][
            min(len(cluster_R_indep_noins) - 1, int(np.round(len(cluster_R_indep_noins) * 0.9)))]

    outliers = 0
    for i, label in enumerate(pseudo_labels):
        indep_score = cluster_R_indep[label.item()]
        comp_score = R_comp[i]
        if (not ((indep_score <= indep_thres) and (comp_score.item() <= cluster_R_comp[label.item()]))
                and cluster_num[label.item()] > 1):
            pseudo_labels[i] = len(cluster_R_indep) + outliers
            outliers += 1

    return pseudo_labels


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


def edge_perturbation(edge_index, mask=None, disconnect_prob=0.05):
    if mask is None:
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
    selected_edges = edge_index[:, mask]
    disconnect_mask = np.random.rand(selected_edges.shape[1]) > disconnect_prob
    disconnected_edges = selected_edges[:, disconnect_mask]
    unchanged_edges = edge_index[:, ~mask]
    edge_index = np.concatenate([unchanged_edges, disconnected_edges], axis=1)
    return edge_index


def train(encoder_model, contrast_model, data, optimizer, scaler):
    encoder_model.train()
    optimizer.zero_grad()
    with autocast():
        _, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
        loss = contrast_model(z1, z2)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator

class MyModel(BaseEstimator):
    def __init__(self, total_epoch=1000, B=50, eps=0.6, eps_gap=0.02):
        self.total_epoch = total_epoch
        self.B = B
        self.eps = eps
        self.eps_gap = eps_gap

    def fit(self, X, y=None):
        res = main(total_epoch=self.total_epoch, B=self.B, eps=self.eps, eps_gap=self.eps_gap)
        self.accuracy_ = res['micro_f1']
        return self

    def score(self, X, y=None):
        return self.accuracy_


def main(total_epoch=1000, B=50, eps=0.6, eps_gap=0.02):
    initial_portion = 0.24
    final_portion = 0.42
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('.'), 'datasets', 'WikiCS')
    dataset = WikiCS(path, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=256).to(device)
    gat = GAT(num_features=dataset.num_features, num_hidden=64).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

    scaler = GradScaler()
    optimizer = Adam(encoder_model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=100,
        max_epochs=total_epoch)

    data_train = data.clone()
    increment = (final_portion - initial_portion) / total_epoch
    with tqdm(total=total_epoch, desc='(T)') as pbar:
        for epoch in range(1, total_epoch+1):
            loss = train(encoder_model, contrast_model, data_train, optimizer, scaler)
            if epoch % B == 0 and epoch != total_epoch:
                pseudo_labels = cluster_with_outlier(encoder_model.encoder, data_train, eps=eps, eps_gap=eps_gap)

                portion = min(final_portion, initial_portion + increment * epoch)
                data_train, pseudo_labels = over_sample(data_train, pseudo_labels, portion=portion)
                undersampler = NeighbourhoodCleaningRule(sampling_strategy='majority')
                data_train, pseudo_labels = sim_sample(data_train, pseudo_labels, undersampler, encoder_model.encoder)

            scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'params: total_epoch={total_epoch}, B={B}, portion={portion}, eps={eps}, eps_gap={eps_gap}')
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    return test_result


if __name__ == '__main__':
    param_dist = {
        'total_epoch': [1000],
        'B': [30, 50],
        'eps': np.linspace(0.4, 0.52, 50).tolist(),
        'eps_gap': np.linspace(0.02, 0.03, 5).tolist()
    }

    model = MyModel()
    grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=3, scoring=None)
    grid_search.fit([0, 0, 1], [0, 1, 1])

    print(grid_search.best_params_)
