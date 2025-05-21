import glob

def get_feature_map(dataset: str) -> list:

    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())
    return feature_list

def get_fc_graph_struc(dataset):
    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)
    return struc_map

def get_prior_graph_struc(dataset):
    feature_file = open(f'./data/{dataset}/features.txt', 'r')
    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if dataset == 'wadi' or dataset == 'wadi2':
                if other_ft is not ft and other_ft[0] == ft[0]:
                    struc_map[ft].append(other_ft)
            elif dataset == 'swat':
                if other_ft is not ft and other_ft[-3] == ft[-3]:
                    struc_map[ft].append(other_ft)
    return struc_map

def get_graph_struc_from_txt(dataset):

    graph = {}
    with open(f'./data/{dataset}/features.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            src, dst = tokens[0], tokens[1]
            if src not in graph:
                graph[src] = []
            graph[src].append(dst)
            if dst not in graph:
                graph[dst] = []
    return graph
