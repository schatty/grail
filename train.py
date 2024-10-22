import sys
import os
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import networkx as nx
import argparse
from tqdm import tqdm
import time
import json

from utils import Logger
from models.mlp import MLPClassifier
from models.transformer import BasicTransformer


def parse_args():
    # TODO: Change command descritption later
    parser = argparse.ArgumentParser(description='Argparser for graph_classification')
    parser.add_argument('--data', default='MUTAG', help='data folder name')
    parser.add_argument('--folds', type=str, default='1', help='fold (1..10)')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--feat_dim', type=int, default=0, help='dimension of node feature')
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of node embedding')
    parser.add_argument('--num_class', type=int, default=0, help='#classes')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='dimension of rnn hidden dimension')
    parser.add_argument('--hidden', type=int, default=64, help='dimension of classification')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='init learning_rate')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda device name")
    parser.add_argument('--name', type=str, default='debug', help="Expeiment name for log folder")
    parser.add_argument('--core', type=str, default='transformer', help="lstm|transformer")
    args = parser.parse_args()
    return args
    

class Graph(object):
    def __init__(self, g, node_tags, label):
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()

        # Add neighbor info
        self.neighbor1 = []
        self.neighbor1_tag = []
        self.neighbor2 = []
        self.neighbor2_tag = []

        for i in range(self.num_nodes):
            self.neighbor1.append(list(g.neighbors(i)))
            self.neighbor1_tag.append([node_tags[w] for w in g.neighbors(i)])
        for i in range(self.num_nodes):
            tmp = []
            for j in self.neighbor1[i]:
                for k in g.neighbors(j):
                    if k != i:
                        tmp.append(k)
            self.neighbor2.append(tmp)
            self.neighbor2_tag.append([node_tags[w] for w in tmp])

        self.adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in self.neighbor1[i]:
                self.adj[i, j] = 1.0
        for i in range(self.num_nodes):
            for j in self.neighbor2[i]:
                self.adj[i, j] = 0.5


def load_data(args, fold):
    g_list = []
    g_neighbor_list = []
    label_dict = {}
    feat_dict = {}

    with open('data/%s/%s.txt' % (args.data, args.data), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [int(w) for w in row]
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            assert len(g.edges()) * 2 == n_edges
            assert len(g) == n
            g_list.append(Graph(g, node_tags, l))
    for g in g_list:
        g.label = label_dict[g.label]
    args.num_class = len(label_dict)
    args.feat_dim = len(feat_dict)
    print('# classes: %d' % args.num_class)
    print('# node features: %d' % args.feat_dim)

    train_idxes = np.loadtxt(f'data/{args.data}/10fold_idx/train_idx-{i_fold}.txt', dtype=np.int32).tolist()
    test_idxes = np.loadtxt(f'data/{args.data}/10fold_idx/test_idx-{i_fold}.txt', dtype=np.int32).tolist()

    return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes], g_list


class GraphClassifier(nn.Module):
    def __init__(self, core, input_size, hidden_size, embedding_size, device):
        super(GraphClassifier, self).__init__()

        self.core = core
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(args.feat_dim, embedding_size)
        if core == "lstm":
            self.model = nn.LSTM(embedding_size*2, hidden_size)
        elif core == "transformer":
            self.model = BasicTransformer(emb=embedding_size*2,
                                          heads=4,
                                          depth=1,
                                          seq_length=40,
                                          num_classes=2)
        else:
            raise ValueError("Unknown graph calssfifier core.")

        # TODO: Remove hardcoding
        if args.core == "lstm":
            hidden = args.hidden
        else:
            hidden = args.hidden * 2

        self.mlp_1 = MLPClassifier(input_size=hidden, hidden_size=args.hidden, num_class=args.num_class)
        self.mlp_2 = MLPClassifier(input_size=hidden, hidden_size=args.hidden, num_class=args.num_class)
        self.mlp_3 = MLPClassifier(input_size=hidden, hidden_size=args.hidden, num_class=args.num_class)
        self.mlp_4 = MLPClassifier(input_size=hidden, hidden_size=args.hidden, num_class=args.num_class)

        self.device = device
        self._update_step = -1

    def forward(self, batch_graph):
        self._update_step += 1

        node_tags = batch_graph[0].node_tags
        node_tags = torch.LongTensor(node_tags).view(-1, 1)
        node_tags = node_tags.to(self.device)
        label = [batch_graph[0].label]
        label = torch.LongTensor(label)
        label = label.to(self.device)
        num_nodes = batch_graph[0].num_nodes
        node_feat = torch.zeros(num_nodes, args.feat_dim)
        node_feat = node_feat.to(self.device)
        node_feat.scatter_(1, node_tags, 1) # turn zero matrix to one-hot
        node_feat = Variable(node_feat)
        node_feat = self.embedding(node_feat)

        # Prepare neighbor features
        neighbor1_tags = batch_graph[0].neighbor1_tag
        adj = batch_graph[0].adj
        neighbor1_feat = Variable(torch.zeros(num_nodes, args.feat_dim))
        for i in range(num_nodes):
            for j in neighbor1_tags[i]:
                neighbor1_feat[i, j] = neighbor1_feat[i, j] + 1.
        neighbor1_feat = neighbor1_feat.to(self.device)

        neighbor1_feat = self.embedding(neighbor1_feat)
        input_feat = torch.cat((node_feat, neighbor1_feat), 1)

        batch = []
        for i_node in range(0, num_nodes):
            node_order = self._get_node_order(i_node, adj, num_nodes)
            node_order = node_order.repeat(1, 128)

            # Swap rows according to the probabilistic order
            node_order = node_order.to(self.device)
            input_feat_perm = torch.gather(input_feat, 0, node_order)
            input_feat_perm = input_feat_perm.view(num_nodes, 1, -1)
            batch.append(input_feat_perm)

        batch = torch.cat(batch, axis=1)
   
        # TODO: remove if logic
        if self.core == "lstm":
            out, hidden = self.model(batch)
            embed = torch.mean(out, dim=0)
        else:
            # Transfomer has dimensitons [batch_size, input_seq, embed]
            batch = batch.permute(1, 0, 2)
            out = self.model(batch, self._update_step)
            embed = torch.mean(out, dim=1)

        return self._ensemble(embed, label)

    def _ensemble(self, embed, label):
        y_1, loss_1, var_w_1 = self.mlp_1(embed, label)
        y_2, loss_2, var_w_2 = self.mlp_2(embed, label)
        y_3, loss_3, var_w_3 = self.mlp_3(embed, label)
        y_4, loss_4, var_w_4 = self.mlp_4(embed, label)

        var_sum = var_w_1 + var_w_2 + var_w_3 + var_w_4
        c1 = var_w_1 / var_sum
        c2 = var_w_2 / var_sum
        c3 = var_w_3 / var_sum
        c4 = var_w_4 / var_sum

        y_ = c1 * y_1 + c2 * y_2 + c3 * y_3 + c4 * y_4
        loss_ = loss_1 + loss_2 + loss_3 + loss_4

        acc = ((y_ >= 0.5).int() == label.item()).float()
        acc = acc.cpu().numpy().item()

        return loss_, acc

    def _get_node_order(self, i_node, adj, num_nodes):
        mask = torch.ones(num_nodes)
        mask[i_node] = 0
        node_order = [i_node] 
        for _ in range(num_nodes - 1):
            prob = torch.from_numpy(adj[i_node, :]).float()
            prob += torch.FloatTensor(num_nodes).uniform_(0.01, 0.1)
            prob = torch.exp(prob)
            prob /= torch.sum(prob)

            # Exclude visited nodes
            prob *= mask
          
            # Find next node to visit
            i_node = torch.argmax(prob).item()
            node_order.append(i_node)
            mask[i_node] = 0

        node_order = torch.LongTensor(node_order).view(-1, 1)
        return node_order

    def init_hidden(self):
        h_t = Variable(torch.zeros(1, self.hidden_size)).to(self.device)
        c_t = Variable(torch.zeros(1, self.hidden_size)).to(self.device)
        return h_t, c_t

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None):
    bsize = 1
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    n_samples = 0
    for pos in range(total_iters):
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        loss, acc = classifier(batch_graph)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()

        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss


if __name__ == '__main__':
    args = parse_args()
    device = args.device

    # Fix seeds
    np.random.seed(0)
    torch.manual_seed(0)
  
    if args.folds == "all":
        folds = [i for i in range(1, 10)]
    else:
        folds = list(map(int, args.folds.split(',')))

    print("CORE: ", args.core)
    
    for i_fold in folds:
        experiment_path = os.path.join("./results", args.name, str(i_fold))
        print("Experiment path: ", experiment_path)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        logger = Logger(experiment_path)

        train_graphs, test_graphs, all_graphs = load_data(args, i_fold)
        print('# train: %d, # test: %d , #total: %d' % (len(train_graphs), len(test_graphs), len(all_graphs)))

        loss_function = nn.NLLLoss()
        input_size = args.feat_dim
        hidden_size = args.rnn_hidden_dim
        embedding_size = args.embedding_dim
        loss_function = nn.NLLLoss()
        classifier = GraphClassifier(args.core, input_size, hidden_size, embedding_size, device)
        classifier = classifier.to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        train_idxes = list(range(len(train_graphs)))
        start = time.time()
        best_loss = None
        print("Running for ", args.num_epochs)
        for epoch in tqdm(range(args.num_epochs)):
            random.shuffle(train_idxes)
            avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
            test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
            
            logger.log_scalar("train_loss", avg_loss[0], epoch)
            logger.log_scalar("train_acc", avg_loss[1], epoch)
            logger.log_scalar("val_loss", test_loss[0], epoch)
            logger.log_scalar("val_acc", test_loss[1], epoch)

        end = time.time()
        print('Time for %d epochs is %.f' %(args.num_epochs, end - start))

        val_loss, val_acc = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        print("Accuracy: ", val_acc)

        metrics_path = os.path.join(experiment_path, "results.txt")
        results = {"val_acc": float(val_acc), "val_loss": float(val_loss)}
        with open(metrics_path, "w") as f:
            json.dump(results, f)
