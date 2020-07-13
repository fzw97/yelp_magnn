import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from pytorchtools import EarlyStopping
from tools import index_generator, parse_minibatch_LastFM, load_yelp_data
from model import MAGNN_lp

bs = 128
num_ntype = 2
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
num_user = 24419
num_item = 27810

no_masks = [[False] * 2, [False] * 2]

etypes_lists = [[[None], [0, 1]],
                [[1, 0], [1, None, 0]]]

use_masks = [[False, True],
             [True, True]]

expected_metapaths = [
    [(0, 0), (0, 1, 0)],
    [(1, 0, 1), (1, 0, 0, 1)]
]


def run_model_yelp(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):

    # adjlists_ua: 各种结点基于metapath的邻接关系，
    # edge_metapath_indices_list_ua: adjlists_ua对应的metapath结点下标序列的集合，
    adjlists_ua, edge_metapath_indices_list_ua, type_mask, \
    train_val_test_pos_user_item, train_val_test_neg_user_item = load_yelp_data()

    device = torch.device('cuda:0')
    features_list = []
    in_dims = []
    if feats_type == 0:  # 按节点类型分别 one-hot初始化feats
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))

    train_pos_user_item = train_val_test_pos_user_item['train_pos_user_item']
    val_pos_user_item = train_val_test_pos_user_item['val_pos_user_item']
    test_pos_user_item = train_val_test_pos_user_item['test_pos_user_item']

    train_neg_user_item = train_val_test_neg_user_item['train_neg_user_item']
    val_neg_user_item = train_val_test_neg_user_item['val_neg_user_item']
    test_neg_user_item = train_val_test_neg_user_item['test_neg_user_item']

    y_true_test = np.array([1] * len(test_pos_user_item) + [0] * len(test_neg_user_item))

    auc_list = []
    ap_list = []
    for _ in range(repeat):
        net = MAGNN_lp(
            num_metapaths_list=[2, 2],
            num_edge_type=2,
            etypes_lists=etypes_lists,
            feats_dim_list=in_dims,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            attn_vec_dim=attn_vec_dim,
            rnn_type=rnn_type,
            dropout_rate=dropout_rate)
        net.to(device)
        print(net)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []

        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_user_item))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_item), shuffle=False)
        num_iter = train_pos_idx_generator.num_iterations()
        print('start training...')
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(num_iter):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()  # (bs,)
                train_pos_idx_batch.sort()
                train_pos_user_item_batch = train_pos_user_item[train_pos_idx_batch].tolist()  # (bs, 2)

                train_neg_idx_batch = np.random.choice(len(train_neg_user_item), len(train_pos_idx_batch))  # (bs,)
                train_neg_idx_batch.sort()
                train_neg_user_item_batch = train_neg_user_item[train_neg_idx_batch].tolist()  # (bs, 2)

                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, train_pos_user_item_batch,
                    device, neighbor_samples, use_masks, num_user)

                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, train_neg_user_item_batch,
                    device, neighbor_samples, no_masks, num_user)

                t1 = time.time()
                dur1.append(t1 - t0)

                [pos_embedding_user, pos_embedding_item], _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_pos_idx_batch_mapped_lists))
                [neg_embedding_user, neg_embedding_item], _ = net(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))

                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_item = pos_embedding_item.view(-1, pos_embedding_item.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_item = neg_embedding_item.view(-1, neg_embedding_item.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_item)
                neg_out = -torch.bmm(neg_embedding_user, neg_embedding_item)

                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:03d}/{:03d} | Iter {:05d}/{:05d} | Train_Loss {:.4f} | PBt {:.4f}s | FFt {:.4f}s | BPt {:.4f}s'
                            .format(epoch, num_epochs, iteration, num_iter, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # validation
            print('start validating...')
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_user_item_batch = val_pos_user_item[val_idx_batch].tolist()
                    val_neg_user_item_batch = val_neg_user_item[val_idx_batch].tolist()
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                        adjlists_ua, edge_metapath_indices_list_ua, val_pos_user_item_batch, device, neighbor_samples, no_masks, num_user)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                        adjlists_ua, edge_metapath_indices_list_ua, val_neg_user_item_batch, device, neighbor_samples, no_masks, num_user)

                    [pos_embedding_user, pos_embedding_item], _ = net(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                    [neg_embedding_user, neg_embedding_item], _ = net(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                    pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                    pos_embedding_item = pos_embedding_item.view(-1, pos_embedding_item.shape[1], 1)
                    neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_item = neg_embedding_item.view(-1, neg_embedding_item.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_user, pos_embedding_item)
                    neg_out = -torch.bmm(neg_embedding_user, neg_embedding_item)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        print('start testing...')
        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_user_item), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_user_item_batch = test_pos_user_item[test_idx_batch].tolist()
                test_neg_user_item_batch = test_neg_user_item[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, test_pos_user_item_batch, device, neighbor_samples, no_masks, num_user)
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, test_neg_user_item_batch, device, neighbor_samples, no_masks, num_user)

                [pos_embedding_user, pos_embedding_item], _ = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
                [neg_embedding_user, neg_embedding_item], _ = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_item = pos_embedding_item.view(-1, pos_embedding_item.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_item = neg_embedding_item.view(-1, neg_embedding_item.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_item).flatten()
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_item).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MAGNN testing for YELP dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=1024, help='Batch size. Default is 1024.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='yelp', help='Postfix for the saved model and result. Default is yelp.')

    args = ap.parse_args()
    run_model_yelp(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)
