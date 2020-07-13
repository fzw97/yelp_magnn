import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle
from tqdm import tqdm

prefix='/data0/wfzdata/datasets/yelp/'


def dict2list():
    prefix = '/data0/lizijian/wechatdata/yelp_processed/'
    train_ui_list = []
    val_ui_list = []
    test_ui_list = []
    with open(prefix + 'train_user_item_dict.pkl', 'rb') as f:
        ui = pickle.load(f)
        keys = list(ui.keys())
        for key in keys:
            for value in ui[key]:
                train_ui_list.append([key, value])

    with open(prefix + 'val_user_item_dict.pkl', 'rb') as f:
        ui = pickle.load(f)
        keys = list(ui.keys())
        for key in keys:
            for value in ui[key]:
                val_ui_list.append([key, value])

    with open(prefix + 'test_user_item_dict.pkl', 'rb') as f:
        ui = pickle.load(f)
        keys = list(ui.keys())
        for key in keys:
            for value in ui[key]:
                test_ui_list.append([key, value])

    train_ui_array = np.array(sorted(train_ui_list), dtype=np.int32)
    val_ui_array = np.array(sorted(val_ui_list), dtype=np.int32)
    test_ui_array = np.array(sorted(test_ui_list), dtype=np.int32)

    np.savez('/data0/wfzdata/datasets/yelp/train_val_test_pos_user_item.npz',
            train_pos_user_item=train_ui_array,
            val_pos_user_item=val_ui_array,
            test_pos_user_item=test_ui_array)


def gen_iuui_idx(prefix='/data0/wfzdata/datasets/yelp/', keeprate=0.1):
    u_u = np.load(prefix + '0/0-0_idx.npy')
    mdata = np.load(prefix + 'adjM.npz')
    adjm = sp.csr_matrix((mdata['data'], mdata['indices'], mdata['indptr']))
    num_user = 24419
    num_item = 27810
    user_item_list = {i: adjm[i, :].nonzero()[1] for i in range(num_user)}

    a_u_u_a = []
    for u1, u2 in tqdm(u_u):
        mpaths = [(a1, u1, u2, a2) for a1 in user_item_list[u1] for a2 in user_item_list[u2]]
        n_mpaths = len(mpaths)
        idx = np.random.choice(np.arange(n_mpaths), int(n_mpaths*keeprate))
        a_u_u_a.extend([mpaths[i] for i in idx])

    a_u_u_a = np.array(a_u_u_a, dtype=np.int32)
    print(a_u_u_a.shape)
    sorted_index = sorted(list(range(len(a_u_u_a))), key=lambda i: a_u_u_a[i, [0, 3, 1, 2]].tolist())
    a_u_u_a = a_u_u_a[sorted_index]

    np.save('/data0/wfzdata/datasets/yelp/0/1-0-0-1_idx.npy', a_u_u_a)


def gen_iuui_adjlist(prefix='/data0/wfzdata/datasets/yelp/'):
    num_user = 24419
    num_item = 27810

    iuui = np.load(prefix + '1/1-0-0-1_idx.npy')
    print(iuui[:2])

    adjdict = dict()
    for i in range(num_item):
        adjdict[i] = '{}'.format(i)

    for i in range(len(iuui)):
        src, _, _, dst = iuui[i]
        adjdict[src-num_user] += ' {}'.format(dst-num_user)

    adjlist = list(adjdict.values())

    with open('/data0/wfzdata/datasets/yelp/1/1-0-0-1.adjlist', 'w') as f:
        for adj in adjlist:
            f.write(adj+'\n')


def neg_sampling(prefix='/data0/wfzdata/datasets/yelp/', neg_ratio=20):
    mdata = np.load(prefix + 'adjM.npz')
    adjm = sp.csr_matrix((mdata['data'], mdata['indices'], mdata['indptr']))
    num_user = 24419
    num_item = 27810
    user_item_list = {i: adjm[i, :].nonzero()[1] for i in range(num_user)}

    neg_ui_list = []
    n_pos = 588122  # 552836 + 17644 + 17642
    train_size = 552836*neg_ratio
    val_size = 17644
    test_size = 17642
    n_neg = train_size + val_size + test_size
    cnt = 0
    while cnt < n_neg:
        if cnt%100000==0:
            print(cnt, '/', n_neg)
        uid = np.random.randint(low=0, high=num_user)
        iid = np.random.randint(low=num_user, high=num_user+num_item)
        if iid in user_item_list[uid]:
            pass
        else:
            cnt += 1
            neg_ui_list.append([uid, iid-num_user])

    train_neg_ui_list = neg_ui_list[:train_size]
    val_neg_ui_list = neg_ui_list[train_size:train_size+val_size]
    test_neg_ui_list = neg_ui_list[train_size+val_size:]

    train_ui_array = np.array(sorted(train_neg_ui_list), dtype=np.int32)
    val_ui_array = np.array(sorted(val_neg_ui_list), dtype=np.int32)
    test_ui_array = np.array(sorted(test_neg_ui_list), dtype=np.int32)

    np.savez('/data0/wfzdata/datasets/yelp/train_val_test_neg_user_item.npz',
            train_neg_user_item=train_ui_array,
            val_neg_user_item=val_ui_array,
            test_neg_user_item=test_ui_array)


def check_data(prefix='/data0/wfzdata/datasets/yelp/'):
    uiu_idx = np.load(prefix + '1/1-0-1_idx.npy')
    print(uiu_idx.shape)
    print(uiu_idx[0])


def idx_npy2dict(prefix='/data0/wfzdata/datasets/yelp/'):

    uu_idx = np.load(prefix + '0/0-0_idx.npy')
    uiu_idx = np.load(prefix + '0/0-1-0_idx.npy')
    iui_idx = np.load(prefix + '1/1-0-1_idx.npy')
    iuui_idx = np.load(prefix + '1/1-0-0-1_idx.npy')

    metapath_indices_mapping = {(0, 0): uu_idx,
                                (0, 1, 0): uiu_idx,
                                (1, 0, 1): iui_idx,
                                (1, 0, 0, 1): iuui_idx}

    expected_metapaths = [
        [(0, 0), (0, 1, 0)],
        [(1, 0, 1), (1, 0, 0, 1)]
    ]

    save_prefix = '/data0/wfzdata/datasets/yelp/'

    num_user = 24419
    num_item = 27810
    target_idx_lists = [np.arange(num_user), np.arange(num_item)]

    for i, metapaths in enumerate(expected_metapaths):
        for metapath in metapaths:
            edge_metapath_idx_array = metapath_indices_mapping[metapath]

            with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
                target_metapaths_mapping = {}
                left = 0
                right = 0
                for target_idx in tqdm(target_idx_lists[i]):
                    while right<len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0]==target_idx:
                        right += 1
                    target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                    left = right
                pickle.dump(target_metapaths_mapping, out_file)


def check_LastFM_data(prefix='/data0/wfzdata/python_workspace/parallel_preprocess/data/preprocessed/LastFM_processed'):
    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    print(idx00)
    print('key=0', idx00[0])
    print('key=233', idx00[233])


def test_npy2idx():
    in_file = open('/data0/wfzdata/datasets/yelp/0/0-1-0_idx.pickle', 'rb')
    iui = pickle.load(in_file)
    in_file.close()

    k = list(iui.keys())
    v = list(iui.values())

    print('k=', len(k))
    print('v=', len(v))

    for key in k:

        print(key, ',', iui[key])
        if key>10:
            break
    pass


def gen_uu_adjlist():
    num_user = 24419
    num_item = 27810

    uu = np.load('/data0/lizijian/wechatdata/yelp_processed/social_network.npy')
    uu = sorted(uu.tolist())

    adjdict = dict()
    for i in range(num_user):
        adjdict[i] = '{}'.format(i)

    for i in range(len(uu)):
        src, dst = uu[i]
        adjdict[src] += ' {}'.format(dst)

    adjlist = list(adjdict.values())

    with open('/data0/wfzdata/datasets/yelp/0/0-0.adjlist', 'w') as f:
        for adj in adjlist:
            f.write(adj+'\n')


def gen_ii_idx():
    num_user = 24419
    num_item = 27810
    iui_idx = np.load(prefix + '1/1-0-1_idx.npy')
    iuui_idx = np.load(prefix + '1/1-0-0-1_idx.npy')
    metapath_indices_mapping = {(1, 0, 1): iui_idx, (1, 0, 0, 1): iuui_idx}
    metapaths = [(1, 0, 1), (1, 0, 0, 1)]
    save_prefix = '/data0/wfzdata/datasets/yelp/'

    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]
        with open(save_prefix + '{}/'.format(1) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in tqdm(range(num_user, num_user+num_item)):
                while right<len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx:
                    right += 1
                target_metapaths_mapping[target_idx-num_user] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)




