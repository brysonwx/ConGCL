from typing import Tuple
import argparse
import os.path as osp
import random
import nni
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, degree, to_undirected, \
    add_self_loops, remove_self_loops
from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE, SugbCon, Pool, Scorer, SugEncoder
from pGRACE.functional import drop_feature, drop_edge_weighted, degree_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, generate_split, Subgraph
from pGRACE.dataset import get_dataset
# from torch.utils.tensorboard import SummaryWriter


def train(param_data, args, epoch):
    model.train()
    optimizer.zero_grad()

    if args.dataset in ('Coauthor-Phy', 'PubMed'):
        idx = torch.randperm(data.x.size(0))[:int(data.x.size(0) * 0.6)].numpy()
        idx_set = {}
        for i, id in enumerate(idx):
            idx_set[id] = i
        x = data.x[idx].cuda(torch.device(args.device))

        data_edge_index = data.edge_index.cpu().numpy()
        row = []
        col = []
        for i in range(data.edge_index.size(1)):
            u, v = data_edge_index[0, i], data_edge_index[1, i]
            if u in idx_set and v in idx_set:
                row.append(idx_set[u])
                col.append(idx_set[v])
        edge_index = torch.stack([torch.Tensor(row), torch.Tensor(col)], dim=0).long().cuda(torch.device(args.device))

        if param['drop_scheme'] == 'degree':
            _drop_weights = degree_drop_weights(edge_index).to(device)
        elif param['drop_scheme'] == 'pr':
            pass
        elif param['drop_scheme'] == 'evc':
            pass
        else:
            _drop_weights = None

        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(edge_index)
            node_deg = degree(edge_index_[1], num_nodes=x.size(0))
            if args.dataset == 'WikiCS':
                _feature_weights = feature_drop_weights_dense(x, node_c=node_deg).to(device)
            else:
                _feature_weights = feature_drop_weights(x, node_c=node_deg).to(device)
        elif param['drop_scheme'] == 'pr':
            pass
        elif param['drop_scheme'] == 'evc':
            pass
        else:
            pass

        def drop_edge(idx: int):
            if param['drop_scheme'] == 'uniform':
                return dropout_adj(edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
            elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(edge_index, _drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)

        x_1 = drop_feature(x, param['drop_feature_rate_1'])
        x_2 = drop_feature(x, param['drop_feature_rate_2'])

        if param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(x, _feature_weights, param['drop_feature_rate_1'])
            x_2 = drop_feature_weighted_2(x, _feature_weights, param['drop_feature_rate_2'])
    else:
        def drop_edge(idx: int):
            global drop_weights

            if param['drop_scheme'] == 'uniform':
                return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
            elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)
        x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

        if param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
            x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    to_device = torch.device(args.device)
    data1 = Data(x=z1, edge_index=edge_index_1).to(to_device, non_blocking=True)
    data2 = Data(x=z2, edge_index=edge_index_2).to(to_device, non_blocking=True)

    # Setting up the subgraph extractor
    sub_model = SugbCon(
        hidden_channels=param_data["num_hidden"],
        encoder=SugEncoder(
            param_data["num_hidden"], param_data["num_proj_hidden"]),
        pool=Pool(in_channels=param_data["num_proj_hidden"]),
        scorer=Scorer(param_data["num_proj_hidden"])).to(to_device)

    ppr_path1 = '{}/{}_1'.format(args.ppr_base_path, args.dataset)
    subgraph1 = Subgraph(data1.x, data1.edge_index, ppr_path1, args.subgraph_size, args.n_order)
    subgraph1.build()
    sample_idx1 = random.sample(range(data1.x.size(0)), data1.x.size(0))
    batch1, index1 = subgraph1.search(sample_idx1)
    zs1, summary1 = sub_model(batch1.x.cuda(device=to_device), batch1.edge_index.cuda(device=to_device), batch1.batch.cuda(device=to_device), index1.cuda(device=to_device))

    ppr_path2 = '{}/{}_2'.format(args.ppr_base_path, args.dataset)
    subgraph2 = Subgraph(data2.x, data2.edge_index, ppr_path2, args.subgraph_size, args.n_order)
    subgraph2.build()
    sample_idx2 = random.sample(range(data2.x.size(0)), data2.x.size(0))
    batch2, index2 = subgraph2.search(sample_idx2)
    zs2, summary2 = sub_model(batch2.x.cuda(device=to_device), batch2.edge_index.cuda(device=to_device), batch2.batch.cuda(device=to_device), index2.cuda(device=to_device))

    _lambda = args.mu
    _gamma = args.gamma
    dataset_name = args.dataset.lower()
    # 1024 if args.dataset in ('ogbn-arxiv', 'Coauthor-Phy') else 
    loss = model.loss(z1, z2, _lambda, dataset_name, epoch, batch_size=None, summary=(summary1, summary2), gamma=_gamma, rm_2sim=args.rm_2sim, rm_alpha=args.rm_alpha)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):
    model.eval()
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    if args.dataset in ('WikiCS'):
        if args.dataset == 'WikiCS':
            accs = []
            for i in range(20):
                acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
                accs.append(acc)
            acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--param', type=str)
    # 39788
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--mu', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--ppr_base_path', type=str)
    parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=5)
    parser.add_argument('--n_order', type=int, help='order of neighbor nodes', default=5)
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=32)
    parser.add_argument('--rm_2sim', type=int, help='rm 2sim for ablation study', default=0)
    parser.add_argument('--rm_alpha', type=int, help='rm alpha for ablation study', default=0)
    # parser.add_argument('--tb_outpath', type=str, default='~/projects/ConGCL/tb_outpath')
                        
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.2,
        'drop_feature_rate_2': 0.3,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
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

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(torch_seed)

    # import os
    # os.makedirs(args.tb_outpath, exist_ok=True)
    # tb_writer = SummaryWriter(args.tb_outpath)

    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

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
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    log = args.verbose.split(',')

    for epoch in range(1, param['num_epochs'] + 1):
        loss = train(param, args, epoch)
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % 100 == 0:
            acc = test()

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

    acc = test(final=True)

    if 'final' in log:
        print(f'{acc}')