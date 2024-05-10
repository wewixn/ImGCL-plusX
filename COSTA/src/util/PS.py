import torch
import numpy as np
import collections
import random
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from imblearn.under_sampling import TomekLinks, NeighbourhoodCleaningRule
from sklearn.cluster import DBSCAN
from torch_geometric.utils import subgraph, to_dense_adj, dense_to_sparse


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
        data4sample = encoder_model(data_clone.x, data_clone.edge_index)
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
    adj_back = adj
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


def cluster_with_outlier(embedding, data, eps=0.6, eps_gap=0.02, min_cluster_size=24):
    device = data.x.device
    data_array = embedding.detach()
    max_values, _ = torch.max(data_array, dim=0)
    min_values, _ = torch.min(data_array, dim=0)
    equal_indices = max_values == min_values
    data_array[:, equal_indices] = 0
    not_equal_indices = torch.logical_not(equal_indices)
    data_array[:, not_equal_indices] = (data_array[:, not_equal_indices] - min_values[not_equal_indices]) / (
                max_values[not_equal_indices] - min_values[not_equal_indices])
    data_array = data_array.cpu().numpy()
    print('euclidean : ', np.linalg.norm(data_array[0] - data_array[1]))


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
