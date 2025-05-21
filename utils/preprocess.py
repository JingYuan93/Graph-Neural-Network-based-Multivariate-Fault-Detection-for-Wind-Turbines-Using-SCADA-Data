import numpy as np
import re

def get_most_common_features(target, all_features, max=3, min=3):
    res = []
    main_keys = target.split('_')
    for feature in all_features:
        if target == feature:
            continue
        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))
        if min <= common_key_num <= max:
            res.append(feature)
    return res

def build_net(target, all_features):
    main_keys = target.split('_')
    edge_indexes = [[], []]
    index_feature_map = [target]
    parent_list = [target]
    graph_map = {}
    depth = 2
    for i in range(depth):
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)
            if feature not in graph_map:
                graph_map[feature] = []
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)
            graph_map[feature] = pure_children
            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)
                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)
        parent_list = pure_children
    return edge_indexes, index_feature_map

def construct_data(data, feature_map, labels=0):

    res = []
    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    sample_n = len(res[0])
    if isinstance(labels, int):
        res.append([labels] * sample_n)
    elif len(labels) == sample_n:
        res.append(labels)
    return res

def build_loc_net(struc, all_features, feature_map=[]):

    index_feature_map = feature_map
    edge_indexes = [[], []]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue
        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue
            c_index = index_feature_map.index(child)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)
    return edge_indexes
