from typing import Any, Optional, Tuple, List
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling
import random
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn.norm.graph_norm import GraphNorm


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.norms = torch.nn.ModuleList()
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            self.norms.append(GraphNorm(2 * out_channels))
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
                self.norms.append(GraphNorm(2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.norms.append(GraphNorm(out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.norms = nn.ModuleList(self.norms)

            self.activation = activation
        else:
            self.norms = nn.ModuleList()
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            self.norms.append(GraphNorm(out_channels))
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
                self.norms.append(GraphNorm(out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.norms = nn.ModuleList(self.norms)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                h = self.conv[i](x, edge_index)
                h = self.norms[i](h)
                x = self.activation(h)
            return x
        else:
            r = self.conv[0](x, edge_index)
            r = self.norms[0](r)
            h = self.activation(r)
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, summary: Tuple, gamma: float, rm_2sim: int):
        f = lambda x: torch.exp(x / self.tau)
        # semantic sim
        s1 = self.sim(z1, z1)
        s2 = self.sim(z1, z2)

        if rm_2sim == 1:
            print('remove the code logic of two sim!')
        else:
            # structural sim
            sum1, sum2 = summary
            ss1 = self.sim(sum1, sum1)
            ss2 = self.sim(sum1, sum2)
            combined_sim1 = gamma * s1 + (1 - gamma) * ss1
            combined_sim2 = gamma * s2 + (1 - gamma) * ss2

            combined_sim3 = gamma * self.sim(z2, z2) + (1 - gamma) * self.sim(sum2, sum2)
            sim1 = torch.cat((combined_sim2.t(), combined_sim3), 1)
            sim2 = torch.cat((combined_sim1, combined_sim2), 1)

            refl_sim = f(combined_sim1)
            between_sim = f(combined_sim2)

            res = -torch.log(
                between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
            return res, sim1, sim2

        refl_sim = f(s1)
        between_sim = f(s2)
        s3 = self.sim(z2, z2)
        sim1 = torch.cat((s2.t(), s3), 1)
        sim2 = torch.cat((s1, s2), 1)
        res = -torch.log(
                between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return res, sim1, sim2

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        print('batched_semi_loss')
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        idx = torch.randperm(num_nodes)
        z1 = z1[idx]
        z2 = z2[idx]
        losses = []
        final_sim1 = None
        final_sim2 = None
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            s1 = self.sim(z1[mask], z1[mask])
            s2 = self.sim(z1[mask], z2[mask])
            refl_sim = f(s1)  # [B, N]
            between_sim = f(s2)  # [B, N]
            losses.append(
                -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())))
        
            # semantic sim
            s3 = self.sim(z2[mask], z2[mask])
            sim1 = torch.cat((s2.t(), s3), 1)
            sim2 = torch.cat((s1, s2), 1)
            if final_sim1 is None and final_sim2 is None:
                final_sim1 = sim1
                final_sim2 = sim2
            else:
                if final_sim1.size(0) == sim1.size(0):
                    final_sim1 = torch.cat((final_sim1, sim1), 1)
                    final_sim2 = torch.cat((final_sim2, sim2), 1)               
            # losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            #                          / (refl_sim.sum(1) + between_sim.sum(1)
            #                             - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        
        return torch.cat(losses), final_sim1, final_sim2

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, _lambda: float, dataset_name: str, epoch: int,
        mean: bool = True, batch_size: Optional[int] = None, summary: Optional[Tuple] = None, gamma: float = 0.9, rm_2sim: int = 0, rm_alpha: int = 0):
        
        sampling = False
        sampling_rate = 1.0
        if dataset_name in ('cora', 'amazon-photo', 'wikics', 'citeseer', 'pubmed'):
            h1 = self.projection(z1)
            h2 = self.projection(z2)
            res_summary = summary
        elif dataset_name == 'coauthor-phy':
            sampling = True
            sampling_rate = 0.6
        elif dataset_name in ('coauthor-cs', 'amazon-computers'):
            sampling = True
            sampling_rate = 0.9
        
        if sampling:
            emb_len = int(z1.size(0) * sampling_rate)
            # emb_inds = torch.randperm(z1.size(0))[0:emb_len]
            sub_z1 = z1[0:emb_len]
            sub_z2 = z2[0:emb_len]
            h1 = self.projection(sub_z1)
            h2 = self.projection(sub_z2)

            sum1, sum2 = summary
            sum11 = sum1[0:emb_len]
            sum22 = sum2[0:emb_len]
            res_summary = (sum11, sum22)

        if batch_size is None:
            l1, sim1, sim2 = self.semi_loss(h1, h2, res_summary, gamma, rm_2sim)
            l2, _sim1, _sim2 = self.semi_loss(h2, h1, res_summary, gamma, rm_2sim)
        else:
            l1, sim1, sim2 = self.batched_semi_loss(h1, h2, batch_size)
            l2, _sim1, _sim2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        if rm_alpha == 1:
            print('remove the code logic of alpha loss!')
        else:
            alpha_1 = -1.0
            alpha_2 = 1.0
            print('alpha-: {}'.format(alpha_1))
            print('alpha+: {}'.format(alpha_2))
            alpha_criterion = AdaptiveAlphaDivLoss(alpha_1, alpha_2, 5.0)
            alpha_loss = alpha_criterion(_sim1, _sim2)
            ret = ret + _lambda * alpha_loss
        return ret


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    assert isinstance(alpha, float)
    q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
    p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()
    q_log_prob = torch.nn.functional.log_softmax(q_logits, dim=1) #gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=1) 
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss


"""
It's often necessary to clip the maximum 
gradient value (e.g., 1.0) when using this adaptive alpha-div loss
"""
class AdaptiveAlphaDivLoss(torch.nn.modules.loss._Loss):
    def __init__(self, alpha_min=-1.0, alpha_max=1.0, iw_clip=5.0):
        super(AdaptiveAlphaDivLoss, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def forward(self, p_dis, q_dis, alpha_min=None, alpha_max=None):
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max
        
        loss_left, grad_loss_left = f_divergence(q_dis, p_dis, alpha_min, iw_clip=self.iw_clip)
        loss_right, grad_loss_right = f_divergence(q_dis, p_dis, alpha_max, iw_clip=self.iw_clip)

        ind = torch.gt(loss_left, loss_right).float()
        loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


"""
class for subgraph pooling
"""
class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):
        super(Pool, self).__init__()
        self.sag_pool = SAGPooling(in_channels, ratio)
        self.lin1 = torch.nn.Linear(in_channels * 2, in_channels)
        
    def forward(self, x, edge, batch, type='mean_pool'):
        if type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)
        elif type == 'sag_pool':
            x1, _, _, batch, _, _ = self.sag_pool(x, edge, batch=batch)
            return global_mean_pool(x1, batch)


class SugbConGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels):
        super(SugbConGCNConv, self).__init__(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        return x


class SugEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SugEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = GCNConv(in_channels, self.hidden_channels) 
        # self.conv = SugbConGCNConv(in_channels, self.hidden_channels)
        self.prelu = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.conv(x, edge_index)
        x1 = self.prelu(x1)
        return x1 


EPS = 1e-15

class SugbCon(torch.nn.Module):

    def __init__(self, hidden_channels, encoder, pool, scorer):
        super(SugbCon, self).__init__()
        self.encoder = encoder
        self.hidden_channels = hidden_channels
        self.pool = pool
        self.scorer = scorer
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.scorer)
        reset(self.encoder)
        reset(self.pool)
        
    def forward(self, x, edge_index, batch=None, index=None):
        r""" Return node and subgraph representations of each node before and after being shuffled """
        hidden = self.encoder(x, edge_index)
        if index is None:
            return hidden
        
        z = hidden[index]
        summary = self.pool(hidden, edge_index, batch)
        return z, summary
    
    
    def loss(self, hidden1, summary1):
        r"""Computes the margin objective."""

        shuf_index = torch.randperm(summary1.size(0))

        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]
        
        logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim = -1))
        logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim = -1))
        
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        
        return TotalLoss


    def test(self, train_z, train_y, val_z, val_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        val_acc = clf.score(val_z.detach().cpu().numpy(), val_y.detach().cpu().numpy())
        test_acc = clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy())
        return val_acc, test_acc


class Scorer(nn.Module):
    def __init__(self, hidden_size):
        super(Scorer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

    def forward(self, input1, input2):
        output = torch.sigmoid(torch.sum(input1 * torch.matmul(input2, self.weight), dim = -1))
        return output