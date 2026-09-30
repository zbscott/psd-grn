

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.preprocessing import label_binarize, StandardScaler
from tensorboardX import SummaryWriter

from torch_geometric_signed_directed.utils import link_class_split
from torch_geometric_signed_directed.data import SignedData


from get_psdgrn_laplacian import build_psdgrn_laplacian
from parser_link import parameter_parser

args = parameter_parser()

try:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATA_ROOT = os.path.abspath(args.data_root) if args.data_root else os.path.join(PROJECT_ROOT, 'Datasets')
OUTPUT_ROOT = os.path.abspath(args.output_root) if args.output_root else PROJECT_ROOT
DATASET_DIR = os.path.join(DATA_ROOT, args.dataset)

EXPR_PATH = os.path.join(DATASET_DIR, args.expression_file)
net_data_path = os.path.join(DATASET_DIR, 'gold_augmented.csv')
clean_gold_path = os.path.join(DATASET_DIR, f'Signed_GT_{args.dataset}_clean.csv')
conflict_gold_path = os.path.join(DATASET_DIR, f'Signed_GT_{args.dataset}_conflicts.csv')

PSEUDOTIME_PATH = os.path.join(DATASET_DIR, 'PseudoTime.csv')

N_BINS = 5

log_path = os.path.join(OUTPUT_ROOT, 'logs', args.method, args.dataset)
if os.path.isdir(log_path) == False:
    os.makedirs(log_path)


def calculate_metrics(test_y, pred, out, num_classes):
    if num_classes != 2:
        raise ValueError(f"当前指标函数仅用于 2C 任务，实际 num_classes={num_classes}")

    if torch.is_tensor(test_y):
        test_y_np = test_y.cpu().numpy().astype(int)
    else:
        test_y_np = np.asarray(test_y).astype(int)

    pred_np = np.asarray(pred).astype(int)
    prob = torch.softmax(out, dim=1).detach().cpu().numpy()

    if prob.shape[1] != 2:
        raise ValueError(f"2C 任务要求模型输出维度为 2，实际 out.shape={out.shape}")


    auc_list = []
    valid_classes = []

    for c in range(num_classes):
        y_true_c = (test_y_np == c).astype(int)
        y_score_c = prob[:, c]

        if len(np.unique(y_true_c)) < 2:
            continue

        valid_classes.append(c)

        try:
            auc_c = metrics.roc_auc_score(y_true_c, y_score_c)
            auc_list.append(auc_c)
        except Exception:
            pass

    auc = float(np.mean(auc_list)) if len(auc_list) > 0 else np.nan


    positive_class = 1
    y_true_pos = (test_y_np == positive_class).astype(int)
    y_score_pos = prob[:, positive_class]

    if len(np.unique(y_true_pos)) < 2:
        aupr = np.nan
    else:
        aupr = float(metrics.average_precision_score(y_true_pos, y_score_pos))

    precision = metrics.precision_score(test_y_np, pred_np, average='macro', zero_division=0)
    aupr_ratio = aupr / precision if (precision != 0 and not np.isnan(aupr)) else np.nan

    f1 = metrics.f1_score(test_y_np, pred_np, average='macro', zero_division=0)
    mcc = metrics.matthews_corrcoef(test_y_np, pred_np)
    recall = metrics.recall_score(test_y_np, pred_np, average='macro', zero_division=0)
    epr = (recall * precision) / (recall + precision + 1e-6)

    present_labels = list(range(num_classes))

    report = metrics.classification_report(
        test_y_np,
        pred_np,
        labels=present_labels,
        zero_division=0
    )
    print(report)

    cm = metrics.confusion_matrix(test_y_np, pred_np, labels=present_labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{i}' for i in present_labels],
        columns=[f'Pred_{i}' for i in present_labels]
    )
    print("混淆矩阵 Confusion Matrix:")
    print(cm_df)

    print("有效AUC类别:", valid_classes)
    print("AUPR计算类别: class 1 / 有边正类")

    return f1, auc, aupr, aupr_ratio, precision, mcc, recall, epr


def read_gold_table(path):
    """
    兼容以下情况：
    1. 有表头：source,target,sign
    2. 有表头：其他名称的源节点列,target,sign
    3. 无表头：三列 source,target,sign
    """
    df_try = pd.read_csv(path)
    cols_lower = [str(c).strip().lower() for c in df_try.columns]
    col_map = {str(c).strip().lower(): c for c in df_try.columns}

    if {'source', 'target', 'sign'}.issubset(set(cols_lower)):
        df = df_try[[col_map['source'], col_map['target'], col_map['sign']]].copy()
        df.columns = ['source', 'target', 'sign']
        return df

    if {'target', 'sign'}.issubset(set(cols_lower)):
        source_cols = [
            c for c in df_try.columns
            if str(c).strip().lower() not in {'target', 'sign'}
        ]
        if len(source_cols) > 0:
            df = df_try[[source_cols[0], col_map['target'], col_map['sign']]].copy()
            df.columns = ['source', 'target', 'sign']
            return df

    df = pd.read_csv(path, header=None, names=['source', 'target', 'sign'])
    return df


def upper_clean_index(values):
    return pd.Index(values).astype(str).str.strip().str.upper()

def load_gene_name_gold_standard(net_data_path, expression_path,
                                 clean_out_path=None, conflict_out_path=None):
    net = read_gold_table(net_data_path)

    net['source'] = net['source'].astype(str).str.strip()
    net['target'] = net['target'].astype(str).str.strip()
    net['sign'] = pd.to_numeric(net['sign'], errors='coerce')

    net = net.dropna(subset=['source', 'target', 'sign'])
    net = net[net['sign'].isin([-1, 1])].copy()

    print(f'原始金标准边数: {len(net)}')

    expr = pd.read_csv(expression_path, header=0, index_col=0)
    expr.index = expr.index.astype(str).str.strip()
    expr.columns = expr.columns.astype(str).str.strip()


    expr_index_upper = upper_clean_index(expr.index)
    expr_columns_upper = upper_clean_index(expr.columns)

    gold_genes = set(net['source']).union(set(net['target']))

    index_overlap = len(gold_genes.intersection(set(expr_index_upper)))
    col_overlap = len(gold_genes.intersection(set(expr_columns_upper)))

    if index_overlap == 0 and col_overlap == 0:
        raise ValueError("金标准基因名和表达矩阵的行名/列名都对不上，请检查命名体系。")

    if index_overlap >= col_overlap:

        gene_names = expr_index_upper.tolist()
        print("检测到表达矩阵格式: gene × cell")
    else:

        gene_names = expr_columns_upper.tolist()
        print("检测到表达矩阵格式: cell × gene")


    gene_names = [str(g).strip().upper() for g in gene_names]
    gene2id = {g: i for i, g in enumerate(gene_names)}
    num_nodes = len(gene_names)

    net['source_id'] = net['source'].map(gene2id)
    net['target_id'] = net['target'].map(gene2id)

    before_map = len(net)
    net = net.dropna(subset=['source_id', 'target_id']).copy()
    after_map = len(net)

    print(f'映射前边数: {before_map}, 映射到表达矩阵后边数: {after_map}')

    net['source_id'] = net['source_id'].astype(int)
    net['target_id'] = net['target_id'].astype(int)
    net['sign'] = net['sign'].astype(int)

    sign_nunique = net.groupby(['source_id', 'target_id'])['sign'].nunique()
    conflict_pairs = sign_nunique[sign_nunique > 1].index.tolist()

    print(f'冲突边对数: {len(conflict_pairs)}')

    if len(conflict_pairs) > 0:
        conflict_pairs_df = pd.DataFrame(conflict_pairs, columns=['source_id', 'target_id'])

        conflict_df = net.merge(
            conflict_pairs_df,
            on=['source_id', 'target_id'],
            how='inner'
        ).sort_values(['source', 'target', 'sign'])

        if conflict_out_path is not None:
            conflict_df[['source', 'target', 'sign']].drop_duplicates().to_csv(
                conflict_out_path, index=False
            )
            print(f'冲突边已保存到: {conflict_out_path}')

        pair_set = set(conflict_pairs)
        keep_mask = [
            (s, t) not in pair_set
            for s, t in zip(net['source_id'].tolist(), net['target_id'].tolist())
        ]
        net = net[keep_mask].copy()

    before_dedup = len(net)
    net = net.drop_duplicates(subset=['source_id', 'target_id', 'sign']).copy()
    after_dedup = len(net)

    print(f'删除冲突后边数: {before_dedup}')
    print(f'去重后边数: {after_dedup}')

    if clean_out_path is not None:
        net[['source', 'target', 'sign']].to_csv(
            clean_out_path, index=False
        )
        print(f'干净金标准已保存到: {clean_out_path}')

    rows = net['source_id'].values
    cols = net['target_id'].values
    values = net['sign'].astype(float).values

    a_sparse = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    data = SignedData(A=a_sparse)

    return data, num_nodes, gene_names, gene2id


def build_one_graph_features(expression_path, pseudotime_path, num_nodes, n_bins=5):
    expr = pd.read_csv(expression_path, header=0, index_col=0)
    expr = expr.apply(pd.to_numeric, errors='coerce').fillna(0.0)


    expr.index = expr.index.astype(str).str.strip().str.upper()
    expr.columns = expr.columns.astype(str).str.strip()

    if expr.shape[0] != num_nodes and expr.shape[1] == num_nodes:
        expr = expr.T

        expr.index = expr.index.astype(str).str.strip().str.upper()
        expr.columns = expr.columns.astype(str).str.strip()

    if expr.shape[0] != num_nodes:
        raise ValueError(
            f"表达矩阵基因数({expr.shape[0]})和图节点数({num_nodes})不一致，请检查方向/节点映射/基因顺序。"
        )

    pt = pd.read_csv(pseudotime_path)

    if pt.shape[1] == 1:
        pt = pt.reset_index()
        pt.columns = ['cell', 'pseudotime']
    else:
        lower_map = {str(c).strip().lower(): c for c in pt.columns}

        cell_col = None
        for cand in ['cell', 'cells', 'barcode', 'barcodes']:
            if cand in lower_map:
                cell_col = lower_map[cand]
                break
        if cell_col is None:
            cell_col = pt.columns[0]

        time_col = None
        for cand in ['pseudotime', 'ptime', 'time']:
            if cand in lower_map:
                time_col = lower_map[cand]
                break
        if time_col is None:
            time_col = pt.columns[1]

        pt = pt[[cell_col, time_col]].copy()
        pt.columns = ['cell', 'pseudotime']

    pt['cell'] = pt['cell'].astype(str).str.strip()
    pt['pseudotime'] = pd.to_numeric(pt['pseudotime'], errors='coerce')
    pt = pt.dropna().drop_duplicates(subset=['cell']).sort_values('pseudotime')

    common_cells = [c for c in pt['cell'].tolist() if c in expr.columns]
    if len(common_cells) == 0:
        raise ValueError("pseudotime 文件里的细胞名和表达矩阵列名一个都对不上，请检查文件。")

    pt = pt[pt['cell'].isin(common_cells)].sort_values('pseudotime')
    sorted_cells = pt['cell'].tolist()
    expr = expr[sorted_cells]

    expr = np.log2(expr + 1.0)

    pseudotime_values = pt['pseudotime'].values.astype(np.float32)
    pseudotime_values = (pseudotime_values - pseudotime_values.min()) / (
        pseudotime_values.max() - pseudotime_values.min() + 1e-8
    )

    n_bins = min(n_bins, len(sorted_cells))
    if n_bins <= 0:
        raise ValueError("有效 pseudotime 细胞数为 0，无法构造动态特征。")

    bins = np.array_split(np.arange(len(sorted_cells)), n_bins)

    bin_feat_list = []

    for idx in bins:
        Xb = expr.iloc[:, idx].values.astype(np.float32)
        tb = pseudotime_values[idx].astype(np.float32)

        mean_b = Xb.mean(axis=1)
        std_b = Xb.std(axis=1)
        max_b = Xb.max(axis=1)
        min_b = Xb.min(axis=1)

        if len(tb) >= 2 and np.var(tb) > 1e-12:
            tb_center = tb - tb.mean()
            Xb_center = Xb - Xb.mean(axis=1, keepdims=True)
            slope_b = (Xb_center @ tb_center) / (np.sum(tb_center ** 2) + 1e-8)
        else:
            slope_b = np.zeros(Xb.shape[0], dtype=np.float32)

        one_bin_feat = np.stack(
            [
                mean_b,
                std_b,
                max_b,
                min_b,
                slope_b
            ],
            axis=1
        )

        bin_feat_list.append(one_bin_feat)

    stage_feat = np.stack(bin_feat_list, axis=1).astype(np.float32)

    flat_stage_feat = stage_feat.reshape(stage_feat.shape[0], -1)
    scaler = StandardScaler()
    flat_stage_feat = scaler.fit_transform(flat_stage_feat)
    stage_feat = flat_stage_feat.reshape(stage_feat.shape[0], n_bins, 5).astype(np.float32)


    node_features = torch.tensor(stage_feat, dtype=torch.float32)

    return node_features

def ensure_query_edges_shape(query_edges):
    if query_edges.dim() != 2:
        raise ValueError(f"query_edges 维度错误: {query_edges.shape}")

    if query_edges.size(0) == 2 and query_edges.size(1) != 2:
        query_edges = query_edges.t().contiguous()
    elif query_edges.size(1) != 2:
        raise ValueError(f"query_edges 应为 [E,2] 或 [2,E]，实际为 {query_edges.shape}")

    return query_edges.long()

def extract_pos_neg_edges(edge_index, edge_weight):
    edge_index = edge_index.detach().cpu()
    edge_weight = edge_weight.detach().cpu().float().view(-1)

    if edge_index.dim() != 2:
        raise ValueError(f"edge_index 维度错误: {edge_index.shape}")

    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()

    if edge_index.size(0) != 2:
        raise ValueError(f"edge_index 应为 [2,E] 或 [E,2]，实际为 {edge_index.shape}")

    pos_mask = edge_weight > 0
    neg_mask = edge_weight < 0

    pos_edges = edge_index[:, pos_mask].t().numpy().astype(np.int64)
    neg_edges = edge_index[:, neg_mask].t().numpy().astype(np.int64)

    if pos_edges.ndim == 1:
        pos_edges = pos_edges.reshape(-1, 2)
    if neg_edges.ndim == 1:
        neg_edges = neg_edges.reshape(-1, 2)

    if pos_edges.size == 0:
        pos_edges = np.empty((0, 2), dtype=np.int64)
    if neg_edges.size == 0:
        neg_edges = np.empty((0, 2), dtype=np.int64)

    return pos_edges, neg_edges

def get_max_node_id_from_edge_tensor(edges):
    if edges is None:
        return -1
    if not torch.is_tensor(edges):
        return -1
    if edges.numel() == 0:
        return -1
    return int(edges.detach().cpu().max().item())

def get_link_data_max_node_id(link_data):
    max_node_id = -1

    for split in list(link_data.keys()):
        split_data = link_data[split]

        if 'graph' in split_data:
            max_node_id = max(
                max_node_id,
                get_max_node_id_from_edge_tensor(split_data['graph'])
            )

        for part in ['train', 'val', 'test']:
            if part in split_data and isinstance(split_data[part], dict):
                if 'edges' in split_data[part]:
                    max_node_id = max(
                        max_node_id,
                        get_max_node_id_from_edge_tensor(split_data[part]['edges'])
                    )

    return max_node_id

def is_link_data_compatible_with_num_nodes(link_data, num_nodes):
    max_node_id = get_link_data_max_node_id(link_data)
    return max_node_id < num_nodes, max_node_id

def _edge_tensor_to_np_edges(edges):
    if not torch.is_tensor(edges):
        edges = torch.tensor(edges, dtype=torch.long)
    edges = edges.detach().cpu().long()
    if edges.dim() != 2:
        raise ValueError(f"edges 维度错误: {edges.shape}")
    if edges.size(0) == 2 and edges.size(1) != 2:
        edges = edges.t().contiguous()
    if edges.size(1) != 2:
        raise ValueError(f"edges 应为 [E,2] 或 [2,E]，实际为 {edges.shape}")
    return edges.numpy().astype(np.int64)


def _sample_strict_directed_negative_edges(num_nodes, num_samples, forbidden_pairs, rng):
    """
    严格有向负采样：负样本候选池为所有 A[i,j] = 0 的有序基因对。
    只排除：
      1) self-loop，即 i == j；
      2) 金标准中已经存在的同向有向边 (i,j)；
      3) 当前已经采过的负样本。

    注意：如果金标准只有 A->B，没有 B->A，那么 (B,A) 不会因为 (A,B) 存在而被排除，
    因此 B->A 可以作为 no-edge 负样本。
    """
    if num_samples <= 0:
        return np.empty((0, 2), dtype=np.int64)

    neg_edges = []
    max_possible = num_nodes * (num_nodes - 1) - len(forbidden_pairs)
    if num_samples > max_possible:
        raise ValueError(
            f"严格有向负采样候选不足: need={num_samples}, available≈{max_possible}. "
            "请减小负采样数量或检查图规模。"
        )

    trials = 0
    max_trials = max(10000, num_samples * 100)

    while len(neg_edges) < num_samples and trials < max_trials:
        batch_size = max(4096, (num_samples - len(neg_edges)) * 4)
        src = rng.randint(0, num_nodes, size=batch_size)
        dst = rng.randint(0, num_nodes, size=batch_size)

        for u, v in zip(src, dst):
            u = int(u)
            v = int(v)
            if u == v:
                continue
            pair = (u, v)
            if pair in forbidden_pairs:
                continue
            forbidden_pairs.add(pair)
            neg_edges.append(pair)
            if len(neg_edges) >= num_samples:
                break

        trials += batch_size


    if len(neg_edges) < num_samples:
        remaining = []
        for u in range(num_nodes):
            for v in range(num_nodes):
                if u == v:
                    continue
                pair = (u, v)
                if pair not in forbidden_pairs:
                    remaining.append(pair)
        need = num_samples - len(neg_edges)
        if len(remaining) < need:
            raise ValueError(
                f"严格有向负采样候选不足: need={num_samples}, got={len(neg_edges) + len(remaining)}"
            )
        rng.shuffle(remaining)
        neg_edges.extend(remaining[:need])
        forbidden_pairs.update(remaining[:need])

    return np.asarray(neg_edges, dtype=np.int64)


def _shuffle_edges_and_labels(pos_edges, neg_edges, rng, device):
    labels_pos = np.ones(len(pos_edges), dtype=np.int64)   
    labels_neg = np.zeros(len(neg_edges), dtype=np.int64) 

    edges = np.concatenate([pos_edges, neg_edges], axis=0)
    labels = np.concatenate([labels_pos, labels_neg], axis=0)

    order = rng.permutation(len(edges))
    edges = edges[order]
    labels = labels[order]

    return (
        torch.tensor(edges, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device)
    )


def replace_existence_negatives_with_strict_directed(link_data, original_edge_index, num_nodes, seed, device):
    """
    保留 link_class_split 产生的正样本划分和训练图，只替换 2C existence 中的负样本。

    原始 torch_geometric_signed_directed 的负采样会先 to_undirected(edge_index)，
    因此如果 A->B 已存在，B->A 通常不会作为负样本。

    本函数改成严格有向负采样：
      负样本候选池 = {(i,j) | i != j 且原始金标准中不存在 i->j}
    因此如果只有 A->B，没有 B->A，则 B->A 可以被采为 no-edge。
    """
    original_edges_np = _edge_tensor_to_np_edges(original_edge_index)
    pos_pair_set = set((int(u), int(v)) for u, v in original_edges_np.tolist() if int(u) != int(v))

    for split in list(link_data.keys()):
        rng = np.random.RandomState(seed + int(split) if str(split).isdigit() else seed)
        used_or_forbidden = set(pos_pair_set)

        for part in ['train', 'val', 'test']:
            old_edges = _edge_tensor_to_np_edges(link_data[split][part]['edges'])


            pos_mask = np.array(
                [(int(u), int(v)) in pos_pair_set for u, v in old_edges],
                dtype=bool
            )

            pos_edges = old_edges[pos_mask]


            neg_count = len(pos_edges)

            strict_neg_edges = _sample_strict_directed_negative_edges(
                num_nodes=num_nodes,
                num_samples=neg_count,
                forbidden_pairs=used_or_forbidden,
                rng=rng
            )

            if len(strict_neg_edges) != len(pos_edges):
                raise RuntimeError(
                    f"{part} 正负样本数量不一致: "
                    f"positive={len(pos_edges)}, negative={len(strict_neg_edges)}"
                )

            new_edges, new_labels = _shuffle_edges_and_labels(
                pos_edges=pos_edges,
                neg_edges=strict_neg_edges,
                rng=rng,
                device=device
            )
            link_data[split][part]['edges'] = new_edges
            link_data[split][part]['label'] = new_labels

        print(
            f"[StrictDirectedNeg][split {split}] "
            f"train label count={torch.bincount(link_data[split]['train']['label'].detach().cpu(), minlength=2).tolist()}, "
            f"val label count={torch.bincount(link_data[split]['val']['label'].detach().cpu(), minlength=2).tolist()}, "
            f"test label count={torch.bincount(link_data[split]['test']['label'].detach().cpu(), minlength=2).tolist()}"
        )

    return link_data


def generate_and_save_link_data(data, task, args, device, save_data_path, save_data_path_dir):
    print("========== DEBUG before link_class_split ==========")
    print("data.edge_index shape:", data.edge_index.shape)

    if hasattr(data, "edge_weight"):
        print("data.edge_weight shape:", data.edge_weight.shape)
        print("edge_weight unique:", torch.unique(data.edge_weight))

    if data.edge_index.numel() == 0 or data.edge_index.size(1) == 0:
        raise ValueError(
            "当前图中没有任何边，不能执行 link_class_split。"
            "请检查金标准 source/target 是否成功匹配到表达矩阵基因名，"
            "以及过滤后 gold_df 是否为空。"
        )


    split_input_data = data
    original_edge_index_for_directed_neg = data.edge_index.detach().cpu()

    if task == "existence":
        edge_index_cpu = data.edge_index.detach().cpu()

        if hasattr(data, "A"):
            num_nodes_for_split = data.A.shape[0]
        else:
            num_nodes_for_split = int(edge_index_cpu.max().item()) + 1

        rows = edge_index_cpu[0].numpy()
        cols = edge_index_cpu[1].numpy()
        values = np.ones(edge_index_cpu.size(1), dtype=np.float32)

        A_binary = sp.coo_matrix(
            (values, (rows, cols)),
            shape=(num_nodes_for_split, num_nodes_for_split)
        )

        split_input_data = SignedData(A=A_binary)

        print("2C existence task: convert signed graph to unsigned directed graph for link_class_split.")
        print("split_input_data.edge_weight unique:", torch.unique(split_input_data.edge_weight))
        print("2C existence task: after link_class_split, replace negatives by strict directed no-edge pairs.")

    link_data = link_class_split(
        split_input_data,
        splits=args.runs,
        task=task,
        prob_val=args.val_ratio,
        prob_test=1 - args.train_ratio - args.val_ratio,
        seed=args.seed,
        device=device
    )

    if task == "existence":
        link_data = replace_existence_negatives_with_strict_directed(
            link_data=link_data,
            original_edge_index=original_edge_index_for_directed_neg,
            num_nodes=num_nodes_for_split,
            seed=args.seed,
            device=device
        )

    if not os.path.isdir(save_data_path_dir):
        try:
            os.makedirs(save_data_path_dir)
        except FileExistsError:
            print(f'Folder exists for {save_data_path_dir}!')

    torch.save(link_data, save_data_path)
    return link_data

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, log_probs, target):
        target = target.long()

        log_pt = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-8, max=1.0)

        ce_loss = -log_pt
        focal_factor = (1.0 - pt) ** self.gamma

        loss = focal_factor * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def laplacian_matmul(L, X):
    if L.is_sparse:
        return torch.sparse.mm(L, X)
    return torch.matmul(L, X)

class PSDGRNStageBlock(nn.Module):
    def __init__(self, hidden_dim, L_norm_real, L_norm_imag, dropout=0.0):
        super().__init__()

        self.L_norm_real = L_norm_real
        self.L_norm_imag = L_norm_imag
        self.K = len(L_norm_real)

        self.real_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(self.K)
        ])

        self.imag_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(self.K)
        ])

        self.bias_real = nn.Parameter(torch.zeros(hidden_dim))
        self.bias_img = nn.Parameter(torch.zeros(hidden_dim))

        self.norm_real = nn.LayerNorm(hidden_dim)
        self.norm_img = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_real, h_img):
        out_real = 0.0
        out_img = 0.0

        for k in range(self.K):
            Lr = self.L_norm_real[k]
            Li = self.L_norm_imag[k]

            prop_real = laplacian_matmul(Lr, h_real) - laplacian_matmul(Li, h_img)
            prop_img = laplacian_matmul(Lr, h_img) + laplacian_matmul(Li, h_real)

            out_real = out_real + self.real_linears[k](prop_real)
            out_img = out_img + self.imag_linears[k](prop_img)

        out_real = out_real + self.bias_real
        out_img = out_img + self.bias_img

        out_real = F.relu(self.norm_real(out_real))
        out_img = F.relu(self.norm_img(out_img))

        out_real = self.dropout(out_real)
        out_img = self.dropout(out_img)

        return out_real, out_img

class PSDGRNLinkPredictor(nn.Module):
    def __init__(
        self,
        num_features,
        stage_feature_dim,
        hidden,
        L_norm_real,
        L_norm_imag,
        label_dim,
        num_stages=5,
        dropout=0.0
    ):
        super().__init__()

        self.num_stages = num_stages
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)

        self.stage_real_proj = nn.ModuleList([
            nn.Linear(num_features + stage_feature_dim, hidden)
            for _ in range(num_stages)
        ])

        self.stage_img_proj = nn.ModuleList([
            nn.Linear(num_features + stage_feature_dim, hidden)
            for _ in range(num_stages)
        ])

        self.stage_blocks = nn.ModuleList([
            PSDGRNStageBlock(
                hidden_dim=hidden,
                L_norm_real=L_norm_real,
                L_norm_imag=L_norm_imag,
                dropout=dropout
            )
            for _ in range(num_stages)
        ])

        self.res_norm_real = nn.ModuleList([
            nn.LayerNorm(hidden)
            for _ in range(num_stages)
        ])

        self.res_norm_img = nn.ModuleList([
            nn.LayerNorm(hidden)
            for _ in range(num_stages)
        ])

        edge_dim = hidden * 8

        self.link_predictor = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, label_dim)
        )

    def forward(self, X_real, X_img, query_edges, one_graph_features):
        device = X_real.device
        one_graph_features = one_graph_features.to(device)

        if one_graph_features.dim() != 3:
            raise ValueError(
                f"one_graph_features 应为 [num_nodes, num_stages, stage_feature_dim]，"
                f"实际为 {one_graph_features.shape}"
            )

        if one_graph_features.size(1) != self.num_stages:
            raise ValueError(
                f"one_graph_features 的 stage 数应为 {self.num_stages}，"
                f"实际为 {one_graph_features.size(1)}"
            )

        if query_edges.dim() != 2:
            raise ValueError(f"query_edges 维度错误: {query_edges.shape}")

        if query_edges.size(0) == 2 and query_edges.size(1) != 2:
            query_edges = query_edges.t().contiguous()

        query_edges = query_edges.long().to(device)

        h_real_prev = None
        h_img_prev = None

        for i in range(self.num_stages):
            stage_feat_i = one_graph_features[:, i, :]

            stage_input_real = torch.cat([X_real, stage_feat_i], dim=1)
            stage_input_img = torch.cat([X_img, stage_feat_i], dim=1)

            h_real_i = self.stage_real_proj[i](stage_input_real)
            h_img_i = self.stage_img_proj[i](stage_input_img)

            h_real_i = F.relu(h_real_i)
            h_img_i = F.relu(h_img_i)

            h_real_i, h_img_i = self.stage_blocks[i](h_real_i, h_img_i)

            if h_real_prev is not None:
                h_real_i = self.res_norm_real[i](h_real_i + h_real_prev)
                h_img_i = self.res_norm_img[i](h_img_i + h_img_prev)

            h_real_prev = h_real_i
            h_img_prev = h_img_i

        h_real = h_real_prev
        h_img = h_img_prev

        src = query_edges[:, 0]
        dst = query_edges[:, 1]


        src_real = h_real[src]
        dst_real = h_real[dst]
        src_img = h_img[src]
        dst_img = h_img[dst]


        z_i = torch.cat(
            [src_real, src_img],
            dim=1
        )

        z_j = torch.cat(
            [dst_real, dst_img],
            dim=1
        )


        edge_repr = torch.cat(
            [
                z_i,
                z_j,
                z_i * z_j,
                torch.abs(z_i - z_j)
            ],
            dim=1
        )

        logits = self.link_predictor(edge_repr)
        return F.log_softmax(logits, dim=1)


def train_psdgrn(X_real, X_img, y, query_edges, one_graph_features):
    model.train()
    out = model(X_real, X_img, query_edges, one_graph_features)
    loss = criterion(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc = metrics.accuracy_score(
        y.detach().cpu(),
        out.max(dim=1)[1].detach().cpu()
    )
    return loss.detach().item(), train_acc

def validate_psdgrn(X_real, X_img, y, query_edges, one_graph_features):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, query_edges, one_graph_features)
        loss = criterion(out, y)
    return loss.detach().item()

def test_psdgrn(X_real, X_img, y, query_edges, num_classes, one_graph_features):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, query_edges, one_graph_features)

    test_y = y.detach().cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()

    test_acc = metrics.accuracy_score(test_y, pred)
    f1_macro = metrics.f1_score(test_y, pred, average='macro', zero_division=0)
    f1_micro = metrics.f1_score(test_y, pred, average='micro', zero_division=0)
    f1, auc, aupr, aupr_ratio, precision, mcc, recall, epr = calculate_metrics(
        test_y, pred, out, num_classes
    )

    return test_acc, f1_macro, f1_micro, f1, auc, aupr, aupr_ratio, precision, mcc, recall, epr


device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')


USED_EXPR_PATH = EXPR_PATH

data, num_nodes, gene_names, gene2id = load_gene_name_gold_standard(
    net_data_path=net_data_path,
    expression_path=USED_EXPR_PATH,
    clean_out_path=clean_gold_path,
    conflict_out_path=conflict_gold_path
)

print("num_nodes =", num_nodes)


one_graph_features = build_one_graph_features(
    USED_EXPR_PATH,
    PSEUDOTIME_PATH,
    num_nodes=num_nodes,
    n_bins=N_BINS
)

print("one_graph_features shape:", one_graph_features.shape)

sub_dir_name = (
    'runs' + str(args.runs) +
    'epochs' + str(args.epochs) +
    '100train_ratio' + str(int(100 * args.train_ratio)) +
    '100val_ratio' + str(int(100 * args.val_ratio)) +
    '1000lr' + str(int(1000 * args.lr)) +
    '1000weight_decay' + str(int(1000 * args.weight_decay)) +
    '100dropout' + str(int(100 * args.dropout))
)

if args.seed != 0:
    sub_dir_name += 'seed' + str(args.seed)

suffix = (
    'K' + str(args.K) +
    '100q' + str(int(100 * args.q)) +
    'hidden' + str(args.hidden)
)

if args.input_unweighted:
    suffix += 'InputUnweighted'
if args.normalization == 'None':
    suffix += 'no_norm'
if args.num_layers != 2:
    suffix += 'num_layers' + str(args.num_layers)

suffix += f'_PSDGRN_2C_Bins{N_BINS}_MeanStdMaxMinSlope_ArticleEdgeConcat'

logs_folder_name = 'runs'
if args.debug:
    args.runs = 2
    args.epochs = 2
    logs_folder_name = 'debug_runs'

short_suffix = (
    'K' + str(args.K) +
    '_q' + str(int(100 * args.q)) +
    '_h' + str(args.hidden) +
    '_bins' + str(N_BINS)
)

log_dir = os.path.join(OUTPUT_ROOT, 'tb_runs', args.dataset, args.method, sub_dir_name, short_suffix)
os.makedirs(log_dir, exist_ok=True)
print('TensorBoard log_dir:', log_dir)
print('TensorBoard log_dir length:', len(log_dir))
writer = SummaryWriter(log_dir=log_dir)

task = "five_class_signed_digraph"
if args.num_classes == 4:
    task = "four_class_signed_digraph"
elif args.num_classes == 3:
    task = "three_class_digraph"
elif args.num_classes == 2:
    task = "existence" # sign

save_data_path_dir = os.path.join(OUTPUT_ROOT, 'data', args.dataset)

if args.direction_only_task:
    save_data_path = os.path.join(
        save_data_path_dir,
        task + '_direction_only' + str(device) + 'seed' + str(args.seed) +
        'split' + str(args.runs) +
        '100val' + str(int(100 * args.val_ratio)) +
        '100train' + str(int(100 * args.train_ratio)) + '.pt'
    )
else:
    save_data_path = os.path.join(
        save_data_path_dir,
        task + str(device) + 'seed' + str(args.seed) +
        'split' + str(args.runs) +
        '100val' + str(int(100 * args.val_ratio)) +
        '100train' + str(int(100 * args.train_ratio)) + '.pt'
    )


save_data_path = save_data_path.replace('.pt', f'_StrictDirectedNeg_nodes{num_nodes}.pt')

if os.path.exists(save_data_path):
    print('Loading existing data splits!')
    link_data = torch.load(open(save_data_path, 'rb'))

    compatible, cached_max_node_id = is_link_data_compatible_with_num_nodes(
        link_data,
        num_nodes
    )

    if not compatible:
        print(
            "检测到缓存 split 与当前图节点数不兼容，重新生成 split。\n"
            f"cached_max_node_id={cached_max_node_id}, num_nodes={num_nodes}, "
            f"cache_path={save_data_path}"
        )

        link_data = generate_and_save_link_data(
            data=data,
            task=task,
            args=args,
            device=device,
            save_data_path=save_data_path,
            save_data_path_dir=save_data_path_dir
        )
else:
    link_data = generate_and_save_link_data(
        data=data,
        task=task,
        args=args,
        device=device,
        save_data_path=save_data_path,
        save_data_path_dir=save_data_path_dir
    )

compatible, cached_max_node_id = is_link_data_compatible_with_num_nodes(
    link_data,
    num_nodes
)

if not compatible:
    raise ValueError(
        "重新生成后的 link_data 仍然与当前 num_nodes 不兼容。\n"
        f"cached_max_node_id={cached_max_node_id}, num_nodes={num_nodes}。\n"
        "请检查表达矩阵筛选、金标准映射和 link_class_split 输入图是否一致。"
    )

print("link_data max node id:", cached_max_node_id)
print("link_data compatible with num_nodes:", compatible)

if args.direction_only_task:
    task += '_direction_only'
    args.num_classes -= 2

start = time.time()
res_array = np.zeros((args.runs, 11))

for split in list(link_data.keys()):
    print("\n" + "=" * 80)
    print(f"Split {split}")
    print("=" * 80)

    edge_index = link_data[split]['graph']
    edge_weight = link_data[split]['weights']

    edge_index_cpu = edge_index.detach().cpu()
    edge_weight_cpu = edge_weight.detach().cpu().float()

    if args.input_unweighted:
        edge_weight_cpu = torch.where(edge_weight_cpu > 0, 1, -1).float()

    query_edges = ensure_query_edges_shape(link_data[split]['train']['edges'])
    query_val_edges = ensure_query_edges_shape(link_data[split]['val']['edges'])
    query_test_edges = ensure_query_edges_shape(link_data[split]['test']['edges'])
    y = link_data[split]['train']['label']
    y_val = link_data[split]['val']['label']
    y_test = link_data[split]['test']['label']

    query_edges = query_edges.to(device)
    query_val_edges = query_val_edges.to(device)
    query_test_edges = query_test_edges.to(device)
    y = y.to(device)
    y_val = y_val.to(device)
    y_test = y_test.to(device)

    class_counts = torch.bincount(y, minlength=args.num_classes).float()
    class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)

    alpha = torch.sqrt(class_counts.sum() / (args.num_classes * class_counts))

    if args.num_classes > 3:
        alpha[1] *= 1.10
        alpha[3] *= 1.10

    alpha = alpha / alpha.mean()

    criterion = MultiClassFocalLoss(
        alpha=alpha.to(device),
        gamma=0.75,
        reduction='mean'
    ).to(device)


    X_real = torch.empty((num_nodes, 0), dtype=torch.float32, device=device)
    X_img = X_real.clone()
    num_input_feat = X_real.size(1)

    train_pos_edges, train_neg_edges = extract_pos_neg_edges(edge_index_cpu, edge_weight_cpu)

    phase_q = np.pi * args.q
    use_norm = not (args.normalization is None or args.normalization == 'None')

    L_real_list, L_imag_list = build_psdgrn_laplacian(
        pos_edges=train_pos_edges,
        neg_edges=train_neg_edges,
        num_nodes=num_nodes,
        K=args.K,
        q=phase_q,
        device=device,
        norm=use_norm
    )

    if args.method == 'PSD-GRN':
        model = PSDGRNLinkPredictor(
            num_features=num_input_feat,
            stage_feature_dim=one_graph_features.size(2),
            hidden=args.hidden,
            L_norm_real=L_real_list,
            L_norm_imag=L_imag_list,
            label_dim=args.num_classes,
            num_stages=N_BINS,
            dropout=args.dropout
        ).to(device)
    else:
        raise ValueError(f"当前代码只整理了PSD-GRN路线，你的 args.method={args.method}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-4
    )

    print("最终节点特征维度:", X_real.shape)
    print("one_graph_features:", one_graph_features.shape)
    print("num_genes:", X_real.shape[0])
    print("stage_feature_dim:", one_graph_features.shape[2])
    print("num_stages:", one_graph_features.shape[1])
    print("train_pos_edges:", train_pos_edges.shape)
    print("train_neg_edges:", train_neg_edges.shape)

    best_val_loss = float('inf')
    early_stopping = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_psdgrn(X_real, X_img, y, query_edges, one_graph_features=one_graph_features)
        scheduler.step()
     
        writer.add_scalar('train_loss_' + str(split), train_loss, epoch)
        if (epoch + 1) % args.checkpoint == 0:
            val_loss = validate_psdgrn(X_real, X_img, y_val, query_val_edges, one_graph_features=one_graph_features)
            if val_loss < best_val_loss:
                early_stopping = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(), log_path + '/model_err' + str(split) + '.t7')
            else:
                early_stopping += 1
            if early_stopping >=args.patience:
                print(f'split{split}, Early Stopping at epoch {epoch}')
                break

    model.load_state_dict(torch.load(log_path + '/model_err' + str(split) + '.t7'))
    accuracy, f1_macro, f1_micro, f1, auc, aupr, aupr_ratio, precision, mcc, recall, epr = test_psdgrn(
        X_real,
        X_img,
        y_test,
        query_test_edges,
        num_classes=args.num_classes,
        one_graph_features=one_graph_features
    )

    print(
        f'Split: {split:02d}, Test_Acc: {accuracy:.4f}, F1 macro: {f1_macro:.4f}, '
        f'F1 micro: {f1_micro:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AUPR: {aupr:.4f}, '
        f'AUPR Ratio: {aupr_ratio:.4f}, Precision: {precision:.4f}, MCC: {mcc:.4f}, '
        f'Recall: {recall:.4f}, EPR: {epr:.4f}.'
    )

    res_array[split] = [
        accuracy,
        f1_macro,
        f1_micro,
        f1,
        auc,
        aupr,
        aupr_ratio,
        precision,
        mcc,
        recall,
        epr
    ]

end = time.time()

if device.type == 'cuda':
    memory_usage = torch.cuda.max_memory_allocated(device) * 1e-6
else:
    memory_usage = 0.0

metric_names = [
    "Accuracy",
    "MacroF1",
    "MicroF1",
    "F1",
    "AUC",
    "AUPR",
    "AUPR_ratio",
    "Precision",
    "MCC",
    "Recall",
    "EPR"
]

res_mean = np.nanmean(res_array, axis=0)
res_std = np.nanstd(res_array, axis=0)

print(
    "{}'s average Accuracy, MacroF1, MicroF1, F1, AUC, AUPR, AUPR_ratio, "
    "Precision, MCC, Recall, EPR: {}".format(args.method, res_mean)
)

print(
    "{}'s std Accuracy, MacroF1, MicroF1, F1, AUC, AUPR, AUPR_ratio, "
    "Precision, MCC, Recall, EPR: {}".format(args.method, res_std)
)

print("{}'s mean ± std:".format(args.method))
for metric_name, mean_value, std_value in zip(metric_names, res_mean, res_std):
    print("{}: {:.4f} ± {:.4f}".format(metric_name, mean_value, std_value))

print(
    "{}'s total training and testing time: {}s, memory usage: {}M.".format(
        args.method,
        end - start,
        memory_usage
    )
)

if args.debug:
    dir_name = os.path.join(OUTPUT_ROOT, 'link_results', task, 'debug', args.dataset)
else:
    dir_name = os.path.join(OUTPUT_ROOT, 'link_results', task, args.dataset)

save_dir = os.path.join(dir_name, sub_dir_name, args.method)
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        print(f'Folder exists for {save_dir}!')

np.save(os.path.join(save_dir, suffix), res_array)
np.save(
    os.path.join(save_dir, 'runtime_memory_' + suffix),
    np.array([end - start, memory_usage], dtype=np.float64)
)

writer.close()