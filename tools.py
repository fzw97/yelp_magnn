import torch
import dgl
import numpy as np
import pickle
from multiprocessing.pool import Pool
from multiprocessing.dummy import Pool as ThreadPool


def load_yelp_data(prefix='/data0/wfzdata/datasets/yelp/'):
    node_types = np.load(prefix + 'node_types.npy')

    # adjm = sp.load_npz(prefix + '/adjM.npz')
    # mdata = np.load(prefix + 'adjM.npz')  # (52,229, 52,229), 0-user = 24419, 1-item = 27810
    # adjm = sp.csr_matrix((mdata['data'], mdata['indices'], mdata['indptr']))

    # uu_idx = np.load(prefix + '0/0-0_idx.npy')  # 110,900
    with open(prefix + '/0/0-0_idx.pickle', 'rb') as f:
        uu_idx = pickle.load(f)

    with open(prefix + '0/0-0.adjlist', 'r') as f:
        adjlist_uu = [line.strip() for line in f]

    # uiu_idx = np.load(prefix + '0/0-1-0_idx.npy')  # 31,294,670
    with open(prefix + '/0/0-1-0_idx.pickle', 'rb') as f:
        uiu_idx = pickle.load(f)

    with open(prefix + '0/0-1-0.adjlist', 'r') as f:
        adjlist_uiu = [line.strip() for line in f]

    # iui_idx = np.load(prefix + '1/1-0-1_idx.npy')  # 36,657,962  unified IDs(iid = n_user + item_id)
    with open(prefix + '/1/1-0-1_idx.pickle', 'rb') as f:
        iui_idx = pickle.load(f)

    with open(prefix + '1/1-0-1.adjlist', 'r') as f:  # item_id should start from 0 !!
        adjlist_iui = [line.strip() for line in f]

    # iuui_idx = np.load(prefix + '1/1-0-0-1_idx.npy')  # 66,201,138
    with open(prefix + '/1/1-0-0-1_idx.pickle', 'rb') as f:
        iuui_idx = pickle.load(f)

    with open(prefix + '1/1-0-0-1.adjlist', 'r') as f:  # item_id should start from 0 !!
        adjlist_iuui = [line.strip() for line in f]

    # train/val/test: 552,836, 17,644, 17,642
    train_val_test_pos_user_item = np.load(prefix + 'train_val_test_pos_user_item.npz')

    # train/val/test:
    train_val_test_neg_user_item = np.load(prefix + 'train_val_test_neg_user_item.npz')

    return [[adjlist_uu, adjlist_uiu], [adjlist_iui, adjlist_iuui]], \
       [[uu_idx, uiu_idx], [iui_idx, iuui_idx]], \
       node_types, train_val_test_pos_user_item, train_val_test_neg_user_item


def parse_adjlist_LastFM(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                if exclude is not None:  # acyclic metapath
                    # exclude = user_artist_batch
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True
                                for u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True
                                for a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]

                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True
                                for u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True
                                for a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]

                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])
        else:
            # no neighbours...
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)

        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))

    # i -> node_id
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}

    edges = list(map(
        lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges
    ))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping



def _parse(input):
    edges = []
    nodes = set()
    result_indices = []

    row, indices, samples, exclude, offset, mode = input
    row_parsed = list(map(int, row.split(' ')))
    nodes.add(row_parsed[0])
    if len(row_parsed) > 1:
        # sampling neighbors
        if samples is None:
            if exclude is not None:  # acyclic metapath
                # exclude = user_artist_batch
                if mode == 0:
                    mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True
                            for u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                else:
                    mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True
                            for a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]

                neighbors = np.array(row_parsed[1:])[mask]
                result_indices.append(indices[mask])
            else:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
        else:
            # undersampling frequent neighbors
            unique, counts = np.unique(row_parsed[1:], return_counts=True)
            p = []
            for count in counts:
                p += [(count ** (3 / 4)) / count] * count
            p = np.array(p)
            p = p / p.sum()
            samples = min(samples, len(row_parsed) - 1)
            sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
            if exclude is not None:
                if mode == 0:
                    mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True
                            for u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                else:
                    mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True
                            for a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]

                neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                result_indices.append(indices[sampled_idx][mask])
            else:
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
    else:
        # no neighbours...
        neighbors = [row_parsed[0]]
        indices = np.array([[row_parsed[0]] * indices.shape[1]])
        if mode == 1:
            indices += offset
        result_indices.append(indices)

    for dst in neighbors:
        nodes.add(dst)
        edges.append((row_parsed[0], dst))

    return nodes, edges, result_indices


def parse_adjlist_LastFM_paralle(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None, mode=None):
    # map
    bs = len(adjlist)
    samples = [samples] * bs
    exclude = [exclude] * bs
    offset = [offset] * bs
    mode = [mode] * bs

    with ThreadPool() as pool:
        res = pool.map(_parse, zip(adjlist, edge_metapath_indices, samples, exclude, offset, mode))

    nodes = set()
    edges = []
    result_indices = []
    for res in list(res):
        node, edge, res_ind = res
        nodes = nodes.union(node)
        edges.extend(edge)
        result_indices.extend(res_ind)

    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(
        lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges
    ))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch_LastFM(adjlists_ua, edge_metapath_indices_list_ua, user_artist_batch, device, samples=None, use_masks=None, offset=None):
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua)):
        # mode_0 = user, mode_1 = artist
        for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
            if use_mask:
                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(
                    adjlist=[adjlist[row[mode]] for row in user_artist_batch],
                    edge_metapath_indices=[indices[row[mode]] for row in user_artist_batch],
                    samples=samples, exclude=user_artist_batch, offset=offset, mode=mode)
            else:
                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch],
                    [indices[row[mode]] for row in user_artist_batch],
                    samples, offset=offset, mode=mode)

            g = dgl.DGLGraph()
            g.add_nodes(num_nodes)
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)

            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in user_artist_batch]))

    return g_lists, result_indices_lists, idx_batch_mapped_lists


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.bs = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter-1)*self.bs:self.iter_counter*self.bs])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.bs))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
