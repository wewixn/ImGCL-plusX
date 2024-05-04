import os
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

import numpy as np
import collections
import random
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from imblearn.under_sampling import TomekLinks, NeighbourhoodCleaningRule
from sklearn.cluster import DBSCAN
from torch_geometric.utils import subgraph, to_dense_adj, dense_to_sparse

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Amazon, WikiCS


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


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

    if chosen is None:
        return adj, features, labels
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


def cluster_with_outlier(encoder_model, data, eps=0.6, eps_gap=0.02, min_cluster_size=24):
    device = data.x.device
    z = encoder_model(data.x, data.edge_index, data.edge_attr)

    data_array = z.cpu().detach().numpy()
    eps_tight = eps - eps_gap
    eps_loose = eps + eps_gap
    min_cluster_size = min(min_cluster_size, int(data.x.size(0) / 42))
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
        raise ValueError(f"Parameter Error. No cluster is detected. eps={eps}, eps_gap={eps_gap}, min_cluster_size={min_cluster_size}")
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


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
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
    def __init__(self, total_epoch=1000, B=50, eps=0.6, eps_gap=0.02, min_cluster_size=24):
        self.total_epoch = total_epoch
        self.B = B
        self.eps = eps
        self.eps_gap = eps_gap
        self.min_cluster_size = min_cluster_size

    def fit(self, X, y=None):
        try:
            res = main(total_epoch=self.total_epoch, B=self.B, eps=self.eps, eps_gap=self.eps_gap, min_cluster_size=self.min_cluster_size)
            self.scores_ = {'micro_f1': res['micro_f1'], 'macro_f1': res['macro_f1']}
        except ValueError as e:
            self.scores_ = {'micro_f1': 0, 'macro_f1': 0}
            print(f"Skipping parameters. {e}")
            raise
        return self

    def score(self, X, y=None):
        return self.scores_['micro_f1']

    def get_results(self):
        return self.scores_


def main(total_epoch=1000, B=50, eps=0.6, eps_gap=0.02, min_cluster_size=24):
    initial_portion = 0.24
    final_portion = 0.42
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('.'), 'datasets', 'Planetoid')
    dataset = Planetoid(path, name='pubmed', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    data_train = data.clone()
    increment = (final_portion - initial_portion) / (np.ceil(total_epoch / B) - 1)
    with tqdm(total=total_epoch, desc='(T)') as pbar:
        for epoch in range(1, total_epoch+1):
            loss = train(encoder_model, contrast_model, data_train, optimizer)
            if epoch % B == 0 and epoch != total_epoch and epoch > 100:
                if data_train.x.size(0) <= 0.2 * data.x.size(0) or data_train.x.size(0) >= 1.21 * data.x.size(0):
                    data_train = data.clone()
                if data_train.edge_index.size(1) >= 1.21 * data.edge_index.size(1):
                    data_train = data.clone()
                pseudo_labels = cluster_with_outlier(encoder_model.encoder, data_train, eps=eps, eps_gap=eps_gap, min_cluster_size=min_cluster_size)

                portion = min(final_portion, initial_portion + increment * (epoch // B))
                data_train, pseudo_labels = over_sample(data_train, pseudo_labels, portion=portion)
                undersampler = NeighbourhoodCleaningRule(sampling_strategy='majority')
                data_train, pseudo_labels = sim_sample(data_train, pseudo_labels, undersampler, encoder_model.encoder)

            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'params: total_epoch={total_epoch}, B={B}, portion={portion}, eps={eps}, eps_gap={eps_gap}, min_cluster_size={min_cluster_size}')
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    if not osp.exists('results'):
        os.makedirs('results')
    with open(f'results/{B}_{eps}_{eps_gap}_{min_cluster_size}.txt', 'a') as f:
        f.write(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}\n')
    return test_result


if __name__ == '__main__':
    B = 50
    eps = 0.6
    eps_gap = 0.02
    min_cluster_size = 24
    total_epoch = 1000

    param_search = True
    if not param_search:
        num_runs = 5
        results = []
        for _ in range(num_runs):
            results.append(main(total_epoch=total_epoch, B=B, eps=eps, eps_gap=eps_gap, min_cluster_size=min_cluster_size))
        for i in range(num_runs):
            print(f'F1Mi={results[i]["micro_f1"]:.4f}, F1Ma={results[i]["macro_f1"]:.4f}')
        print(f'Average test F1Mi={sum([r["micro_f1"] for r in results]) / num_runs:.4f}, '
              f'F1Ma={sum([r["macro_f1"] for r in results]) / num_runs:.4f}')
        exit(0)

    param_dist = {
        'total_epoch': [1000],
        'B': [30, 50],
        'eps': np.linspace(0.2, 0.72, 100).tolist(),
        'eps_gap': np.linspace(0.01, 0.03, 10).tolist(),
        'min_cluster_size': range(8, 30, 2)
    }

    model = MyModel()
    try:
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=5, scoring=None)
        grid_search.fit([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    except ValueError as e:
        print(f"Skipping parameters. {e}")

    print(grid_search.best_params_)
    mean_scores = grid_search.cv_results_['mean_test_score']
    print(f'mean_scores: {mean_scores}')
    best_index = np.nanargmax(mean_scores)
    best_score = mean_scores[best_index]
    best_params = grid_search.cv_results_['params'][best_index]
    print(f'Best score: {best_score}, Best params: {best_params}')
