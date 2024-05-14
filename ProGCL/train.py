import numpy as np
import argparse
import os.path as osp
import random
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected
from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE, BetaMixture1D 
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
import logging
from pGRACE.PS import sim_sample, over_sample, cluster_with_outlier
from imblearn.under_sampling import TomekLinks, NeighbourhoodCleaningRule


LOG_FORMAT = "%(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='Accuracy.txt',level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

def train(epoch, bmm_model, data_train, model, optimizer, feature_weights, drop_weights):
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data_train.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data_train.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')
    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)

    x_1 = drop_feature(data_train.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data_train.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data_train.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data_train.x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1, z2, epoch, args, bmm_model, batch_size=512 if args.dataset == 'Coauthor-Phy' else None)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data_test, dataset, split, final=False):
    model.eval()
    z = model(data_test.x, data_test.edge_index)
    nclass = dataset[0].y.max().item() + 1
    evaluator = MulticlassEvaluator(n_clusters=nclass, random_state=0, n_jobs=8)
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']
    return acc


def main(B, eps, eps_gap, min_cluster_size):
    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)
    device = torch.device(args.device)
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    y = dataset[0].y.view(-1).numpy()
    data = dataset[0]
    data = data.to(device)
    # generate split
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)

    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    log = args.verbose.split(',')
    alphas_init = torch.tensor([1, 2],dtype=torch.float64).to(device)
    betas_init = torch.tensor([2, 1],dtype=torch.float64).to(device)
    weights_init = torch.tensor([1-args.weight_init, args.weight_init], dtype=torch.float64).to(device)
    bmm_model = BetaMixture1D(args.iters, alphas_init, betas_init, weights_init)

    # B = param.get("B", 30)
    # eps = param.get("eps", 0.336969696969697)
    # eps_gap = param.get("eps_gap", 0.06)
    # min_cluster_size = param.get("min_cluster_size", 26)
    initial_portion = param.get("initial_portion", 0.24)
    final_portion = param.get("final_portion", 0.42)
    increment = (final_portion - initial_portion) / (np.ceil(param['num_epochs'] / B) - 1)
    data_train = data.clone()
    data_test = data_train.clone()

    for epoch in range(1, param['num_epochs'] + 1):
        loss = train(epoch, bmm_model, data_train, model, optimizer, feature_weights, drop_weights)
        if 'train' in log and epoch % 100 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % B == 0 and epoch != param['num_epochs'] and epoch > 200:
            if data_train.x.size(0) <= 0.2 * data_test.x.size(0) or data_train.x.size(0) >= 1.21 * data_test.x.size(0):
                data_train = data_test.clone()
            if data_train.edge_index.size(1) >= 1.21 * data_test.edge_index.size(1):
                data_train = data_test.clone()
            pseudo_labels = cluster_with_outlier(model, data_train, eps=eps, eps_gap=eps_gap, min_cluster_size=min_cluster_size)
            portion = min(final_portion, initial_portion + increment * (epoch // B))
            data_train, pseudo_labels = over_sample(data_train, pseudo_labels, portion=portion)
            undersampler = NeighbourhoodCleaningRule(sampling_strategy='majority')
            data_train, pseudo_labels = sim_sample(data_train, pseudo_labels, undersampler, model)
            print('data:', data_train)
            if param['drop_scheme'] == 'degree':
                drop_weights = degree_drop_weights(data_train.edge_index).to(device)
            elif param['drop_scheme'] == 'pr':
                drop_weights = pr_drop_weights(data_train.edge_index, aggr='sink', k=200).to(device)
            elif param['drop_scheme'] == 'evc':
                drop_weights = evc_drop_weights(data_train).to(device)
            else:
                drop_weights = None

        if epoch % 100 == 0:
            acc = test(model, data_test, dataset, split)
            logging.info('\t%.4f'%(acc))
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

    acc = test(model, data_test, dataset, split, final=True)
    logging.info('Final:')
    logging.info('\t%.4f'%(acc))
    print(f'params: B={B}, eps={eps}, eps_gap={eps_gap}, min_cluster_size={min_cluster_size}')

    if 'final' in log:
        print(f'{acc}')
    return acc


from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator

class MyModel(BaseEstimator):
    def __init__(self, B=50, eps=0.6, eps_gap=0.02, min_cluster_size=24):
        self.B = B
        self.eps = eps
        self.eps_gap = eps_gap
        self.min_cluster_size = min_cluster_size
        self.scores_ = 0

    def fit(self, X, y=None):
        try:
            self.scores_ = main(self.B, self.eps, self.eps_gap, self.min_cluster_size)
        except ValueError as e:
            self.scores_ = 0
            print(f"Skipping parameters. {e}")
            raise
        return self

    def score(self, X, y=None):
        return self.scores_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Amazon-Computers')
    parser.add_argument('--param', type=str, default='local:amazon_computers.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--epoch_start', type=int, default=400)
    parser.add_argument('--mode', type=str, default='weight')
    parser.add_argument('--sel_num', type=int, default=1000)
    parser.add_argument('--weight_init', type=float, default=0.05)
    parser.add_argument('--iters', type=int, default=10)
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'evc',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    param_dist = {
        'B': [30, 50],
        'eps': np.linspace(0.42, 2, 200).tolist(),
        'eps_gap': np.linspace(0.24, 0.42, 20).tolist(),We taps into the potential of the GNN encoders that can capture the intricate relationships and dependencies between nodes in the graph, amalgamating nodes and their contextual neighbourhood information as features sampled.将上述这句话重新生成，使之更专业化与学术化。
        'min_cluster_size': range(8, 30, 2)
    }

    model = MyModel()
    try:
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=3, scoring=None)
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