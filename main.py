import torch
from models import GrCNet, GrCNetConvOnly
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from preprocess import  read_entity_from_id, read_relation_from_id,  init_embeddings, build_data, Corpus, save_model
import random
import argparse
import os
import sys
import logging
import time
import pickle
import gc
import subprocess

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# 设置随机种子
set_seed(42)
print("随机种子已设置为 42")

CUDA_LAUNCH_BLOCKING = 1


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/umls", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=0.00001, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=True, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1.5e-3)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/umls/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=128, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=1, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.3, help="Dropout probability for convolution layer")
    # 默认启动多头注意力筛选，但会根据平均入度自动调整
    args.add_argument("-head_sel_ratio", "--head_selection_ratio", type=float,
                      default=1, help="Ratio of heads to keep during selection")

    # 随机种子参数
    args.add_argument("-seed", "--seed", type=int,
                      default=42, help="Random seed for reproducibility")

    # 相似度计算方法选择参数
    args.add_argument("-sim_method", "--similarity_method", type=str,
                      default="jaccard", choices=["jaccard", "cosine"],
                      help="Similarity calculation method: jaccard or cosine")
    args = args.parse_args()

    set_seed(args.seed)
    print(f"随机种子已设置为 {args.seed}")

    return args


args = parse_args()



def load_data(args):
    """
    数据加载函数，支持大规模数据集，考虑出边和入边的分层概念相似性
    """
    import time
    start_time = time.time()

    # 1. 加载基础数据
    train_triples, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        args.data, is_unweigted=False, directed=True)

    num_entities = len(entity2id)
    num_relations = len(relation2id)
    print(f"数据统计: {num_entities} 个实体, {num_relations} 个关系")

    # 验证数据索引范围
    max_entity_idx = max([max(triple[0], triple[2]) for triple in train_triples])
    max_relation_idx = max([triple[1] for triple in train_triples])

    print(f"训练数据索引范围 - 实体: 0-{max_entity_idx}, 关系: 0-{max_relation_idx}")
    print(f"有效索引范围 - 实体: 0-{num_entities - 1}, 关系: 0-{num_relations - 1}")

    if max_entity_idx >= num_entities:
        print(f"警告: 实体索引超出范围!")
    if max_relation_idx >= num_relations:
        print(f"警告: 关系索引超出范围!")

    # 计算平均实体入度
    print("计算平均实体入度...")
    in_degree_counter = {}
    for triple in train_triples:
        tail_entity = triple[2]  # 尾实体索引
        in_degree_counter[tail_entity] = in_degree_counter.get(tail_entity, 0) + 1
    
    # 计算平均入度
    total_in_degree = sum(in_degree_counter.values())
    avg_in_degree = total_in_degree / num_entities if num_entities > 0 else 0
    
    print(f"图谱统计信息:")
    print(f"   - 总实体数: {num_entities}")
    print(f"   - 总入度数: {total_in_degree}")
    print(f"   - 平均实体入度: {avg_in_degree:.2f}")
    
    # 根据平均入度自动决定是否使用头筛选
    if avg_in_degree < 10:
        args.use_head_selection = False
        print(f"平均实体入度 < 10，自动禁用头筛选功能")
    else:
        args.use_head_selection = True
        print(f"平均实体入度 >= 10，自动启用头筛选功能")
    
    # 打印入度分布信息
    max_in_degree = max(in_degree_counter.values()) if in_degree_counter else 0
    min_in_degree = min(in_degree_counter.values()) if in_degree_counter else 0
    print(f"   - 最大实体入度: {max_in_degree}")
    print(f"   - 最小实体入度: {min_in_degree}")
    
    # 计算入度为0的实体比例
    zero_in_degree_entities = num_entities - len(in_degree_counter)
    zero_in_degree_ratio = zero_in_degree_entities / num_entities if num_entities > 0 else 0
    print(f"   - 入度为0的实体比例: {zero_in_degree_ratio:.2%}")

    # 2. 加载嵌入
    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(
            os.path.join(args.data, 'entity2vec.txt'),
            os.path.join(args.data, 'relation2vec.txt'))
        print("从 TransE 初始化关系和实体嵌入")
    else:
        # 使用更高效的随机初始化
        entity_embeddings = np.random.randn(num_entities, args.embeding_size).astype(np.float32)
        relation_embeddings = np.random.randn(num_relations, args.embedding_size).astype(np.float32)
        print("随机初始化关系和实体嵌入")

    emb_time = time.time()
    print(f"嵌入加载耗时: {emb_time - start_time:.2f}秒")

    # 3. 构建分层布尔矩阵（考虑出边和入边）
    matrix_file_path = os.path.join(args.data, 'entity_2*relation_boolean_matrix.npy')

    if not os.path.exists(matrix_file_path):
        print("构建实体-关系分层布尔矩阵...")
        
        # 形式背景的维度：实体数 × (2 × 关系数)
        # 前num_relations列对应r_out，后num_relations列对应r_in
        boolean_matrix = np.zeros((num_entities, 2 * num_relations), dtype=np.bool_)
        
        # 处理每个训练三元组
        for triple in train_triples:
            h_idx, r_idx, t_idx = triple
            
            # 头实体具有r_out属性
            boolean_matrix[h_idx, r_idx] = True  # r_out在矩阵的前半部分
            
            # 尾实体具有r_in属性  
            boolean_matrix[t_idx, r_idx + num_relations] = True  # r_in在矩阵的后半部分
        
        np.save(matrix_file_path, boolean_matrix.astype(np.int32))
        print(f"保存实体-关系布尔矩阵到 {matrix_file_path}")
        print(f"布尔矩阵形状: {boolean_matrix.shape} (实体数: {num_entities}, 属性数: {2 * num_relations})")
    else:
        print(f"加载现有布尔矩阵: {matrix_file_path}")
        boolean_matrix = np.load(matrix_file_path)
        print(f"布尔矩阵形状: {boolean_matrix.shape}")

    bool_matrix_time = time.time()
    print(f"布尔矩阵处理耗时: {bool_matrix_time - emb_time:.2f}秒")

    # 4. 计算双向相似度矩阵并融合
    A_bool = boolean_matrix.astype(np.float32)  # 转为浮点方便计算

    # 4.1 分别计算出边和入边的相似度
    similarity_method = getattr(args, 'similarity_method', 'jaccard')  # 默认为Jaccard
    
    # 计算每个实体在出边和入边上的关系数
    row_sum_out = A_bool[:, :num_relations].sum(axis=1)  # 出边关系数
    row_sum_in = A_bool[:, num_relations:].sum(axis=1)   # 入边关系数
    row_sum_total = row_sum_out + row_sum_in              # 总关系数

    # 对于大规模数据集，直接计算分层Q矩阵而不存储完整的S矩阵
    Q_file = os.path.join(args.data, f'granular_weight_Q_{similarity_method}_layered.npy')

    if not os.path.exists(Q_file):
        print(f"使用分层概念相似性计算粒度权重矩阵 Q ({similarity_method}相似度)...")

        # 检查数据集规模，决定是否使用分块计算
        if num_entities > 10000:  # 大规模数据集
            print(f"检测到大规模数据集 ({num_entities} 个实体)，使用分块计算...")
            Q = _compute_Q_layered_large_scale(A_bool, row_sum_out, row_sum_in, row_sum_total, 
                                             similarity_method, num_entities, num_relations)
        else:  # 小规模数据集
            print("使用常规计算...")
            Q = _compute_Q_layered_normal(A_bool, row_sum_out, row_sum_in, row_sum_total,
                                        similarity_method, num_entities, num_relations)

        np.save(Q_file, Q)
        print(f"保存分层粒度权重矩阵 Q 到 {Q_file}")

        # 打印Q矩阵的统计信息
        print(f"分层权重矩阵Q统计: 最小值={Q.min():.4f}, 最大值={Q.max():.4f}, 均值={Q.mean():.4f}")
        print(f"权重矩阵Q形状: {Q.shape}")

    else:
        print(f"加载现有分层粒度权重矩阵: {Q_file}")
        Q = np.load(Q_file)

    Q_time = time.time()
    print(f"权重矩阵计算耗时: {Q_time - bool_matrix_time:.2f}秒")

    print(f"最终权重矩阵形状: {Q.shape} (实体×关系)")

    # 5. 创建语料库
    print("创建数据语料库...")
    corpus = Corpus(
        args, train_data, validation_data, test_data, entity2id, relation2id,
        headTailSelector, args.batch_size_gat, args.valid_invalid_ratio_gat,
        unique_entities_train, granularity_matrix=Q
    )

    total_time = time.time() - start_time
    print(f"数据加载总耗时: {total_time:.2f}秒")

    # 将嵌入转移到GPU
    entity_tensor = torch.FloatTensor(entity_embeddings)
    relation_tensor = torch.FloatTensor(relation_embeddings)

    if torch.cuda.is_available():
        entity_tensor = entity_tensor.cuda()
        relation_tensor = relation_tensor.cuda()

    return corpus, entity_tensor, relation_tensor


def _compute_layered_similarity_block(A_bool, row_sum_out, row_sum_in, row_sum_total, 
                                    similarity_method, entity_indices, full_entity_indices=None):
    """
    计算实体块的双向相似度（出边相似度 + 入边相似度）
    """
    if full_entity_indices is None:
        full_entity_indices = np.arange(A_bool.shape[0])
    
    num_relations = A_bool.shape[1] // 2
    
    # 提取出边和入边的布尔矩阵
    A_out = A_bool[:, :num_relations]
    A_in = A_bool[:, num_relations:]
    
    # 提取当前块的实体关系向量
    A_out_block = A_out[entity_indices, :]
    A_in_block = A_in[entity_indices, :]
    
    # 计算出边相似度
    intersection_out = A_out_block @ A_out[full_entity_indices, :].T
    
    # 计算入边相似度  
    intersection_in = A_in_block @ A_in[full_entity_indices, :].T
    
    if similarity_method == 'jaccard':
        # 出边Jaccard相似度
        row_sum_out_block = row_sum_out[entity_indices]
        row_sum_out_full = row_sum_out[full_entity_indices]
        union_out = row_sum_out_block[:, np.newaxis] + row_sum_out_full - intersection_out
        union_out_safe = np.where(union_out == 0, 1.0, union_out)
        S_out = intersection_out / union_out_safe
        
        # 入边Jaccard相似度
        row_sum_in_block = row_sum_in[entity_indices]
        row_sum_in_full = row_sum_in[full_entity_indices]
        union_in = row_sum_in_block[:, np.newaxis] + row_sum_in_full - intersection_in
        union_in_safe = np.where(union_in == 0, 1.0, union_in)
        S_in = intersection_in / union_in_safe
        
    elif similarity_method == 'cosine':
        # 出边余弦相似度
        row_sum_out_block = row_sum_out[entity_indices]
        row_sum_out_full = row_sum_out[full_entity_indices]
        row_sum_out_block_safe = np.where(row_sum_out_block == 0, 1.0, row_sum_out_block)
        row_sum_out_full_safe = np.where(row_sum_out_full == 0, 1.0, row_sum_out_full)
        denom_out = np.sqrt(np.outer(row_sum_out_block_safe, row_sum_out_full_safe))
        S_out = intersection_out / denom_out
        
        # 入边余弦相似度
        row_sum_in_block = row_sum_in[entity_indices]
        row_sum_in_full = row_sum_in[full_entity_indices]
        row_sum_in_block_safe = np.where(row_sum_in_block == 0, 1.0, row_sum_in_block)
        row_sum_in_full_safe = np.where(row_sum_in_full == 0, 1.0, row_sum_in_full)
        denom_in = np.sqrt(np.outer(row_sum_in_block_safe, row_sum_in_full_safe))
        S_in = intersection_in / denom_in
        
    else:
        raise ValueError(f"不支持的相似度计算方法: {similarity_method}")
    
    # 融合出边和入边相似度：取平均值
    S_combined = (S_out + S_in) / 2.0
    
    # 确保无效行为零
    valid_rows_block = row_sum_total[entity_indices] > 0
    valid_rows_full = row_sum_total[full_entity_indices] > 0
    valid_mask = np.outer(valid_rows_block, valid_rows_full)
    S_combined[~valid_mask] = 0
    
    return S_combined


def _compute_Q_layered_normal(A_bool, row_sum_out, row_sum_in, row_sum_total, 
                            similarity_method, num_entities, num_relations):
    """
    常规计算分层Q矩阵（适合小规模数据集）
    """
    print("使用分层常规计算方式...")
    
    # 计算完整的分层相似度矩阵 S
    S = _compute_layered_similarity_block(A_bool, row_sum_out, row_sum_in, row_sum_total,
                                        similarity_method, np.arange(num_entities), np.arange(num_entities))
    
    # 分别计算出边和入边的贡献，然后合并
    A_out = A_bool[:, :num_relations]  # 出边部分
    A_in = A_bool[:, num_relations:]   # 入边部分
    
    # 计算出边权重
    numerator_out = S @ A_out
    # 计算入边权重  
    numerator_in = S @ A_in
    
    # 合并出边和入边的权重（取平均值或其他融合策略）
    numerator_combined = (numerator_out + numerator_in) / 2.0
    
    denominator = S.sum(axis=1)
    denominator_safe = np.where(denominator == 0, 1.0, denominator)
    
    # 计算最终的Q矩阵
    Q = numerator_combined / denominator_safe[:, np.newaxis]
    
    return Q


def _compute_Q_layered_large_scale(A_bool, row_sum_out, row_sum_in, row_sum_total,
                                 similarity_method, num_entities, num_relations, block_size=1000):
    """
    大规模数据集的分块计算分层Q矩阵
    """
    print(f"使用分层分块计算方式，块大小: {block_size}")
    
    # 初始化Q矩阵
    Q = np.zeros((num_entities, num_relations), dtype=np.float32)
    
    # 分块计算
    num_blocks = (num_entities + block_size - 1) // block_size
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, num_entities)
        entity_indices = np.arange(start_idx, end_idx)
        
        print(f"处理块 {block_idx + 1}/{num_blocks} (实体 {start_idx}-{end_idx - 1})")
        
        # 计算当前块与所有实体的分层相似度
        S_block = _compute_layered_similarity_block(A_bool, row_sum_out, row_sum_in, row_sum_total,
                                                  similarity_method, entity_indices, np.arange(num_entities))
        
        # 分别处理出边和入边
        A_out = A_bool[:, :num_relations]
        A_in = A_bool[:, num_relations:]
        
        numerator_out_block = S_block @ A_out
        numerator_in_block = S_block @ A_in
        numerator_combined_block = (numerator_out_block + numerator_in_block) / 2.0
        
        denominator_block = S_block.sum(axis=1)
        denominator_safe_block = np.where(denominator_block == 0, 1.0, denominator_block)
        
        Q_block = numerator_combined_block / denominator_safe_block[:, np.newaxis]
        Q[entity_indices] = Q_block
        
        # 释放内存
        del S_block, numerator_out_block, numerator_in_block, numerator_combined_block, Q_block
        
    return Q

Corpus_, entity_embeddings, relation_embeddings = load_data(args)

entity_embeddings_copied = deepcopy(entity_embeddings).cuda()
relation_embeddings_copied = deepcopy(relation_embeddings).cuda()

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))

CUDA = torch.cuda.is_available()
print(CUDA)


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def train_gat(args):
    # 创建输出文件夹
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    print("Defining model with head selection")
    print(
        f"Model type -> GAT layer with {args.nheads_GAT[0]} heads, Head selection: {args.use_head_selection}, Ratio: {args.head_selection_ratio}")

    model_gat = GrCNet(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT,
                                use_head_selection=args.use_head_selection,
                                head_selection_ratio=args.head_selection_ratio)
    if CUDA:
        model_gat.cuda()

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    train_log_file = f"{args.output_folder}/gat_train_log.txt"
     
    print("Starting GAT training from scratch")
    with open(train_log_file, "w") as f:
        f.write("Iteration,Iteration_time,Iteration_loss,Epoch,Average_loss,Epoch_time\n")
    start_epoch = 0

    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(start_epoch, args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                                          len(Corpus_.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()
            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed, _ = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices)
            # 性能监控
            if args.use_head_selection and iters % 50 == 0:  # 每50个迭代监控一次
                analysis_data = model_gat.get_attention_analysis_data()
                if analysis_data and 'computation_savings' in analysis_data:
                    savings = analysis_data['computation_savings']
                    print(f"头选择统计: {savings['selected_heads']}/{savings['total_heads']} 个头被选中 "
                          f"(计算量减少 {savings['computation_reduced']:.1f}%)")
                # 记录到日志文件
                with open(train_log_file, 'a') as f:
                    if analysis_data and 'computation_savings' in analysis_data:
                        savings = analysis_data['computation_savings']
                        f.write(
                            f"Head Selection -> Iteration {iters}: {savings['selected_heads']}/{savings['total_heads']} heads, "
                            f"Reduction: {savings['computation_reduced']:.1f}%\n")

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))
            with open(train_log_file, 'a') as f:
                f.write("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}\n".format(
                    iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_time = time.time() - start_time

        # 性能监控：
        if args.use_head_selection:
            analysis_data = model_gat.get_attention_analysis_data()
            if analysis_data and 'computation_savings' in analysis_data:
                savings = analysis_data['computation_savings']
                print(f"Epoch {epoch} 头选择总结: {savings['selected_heads']}/{savings['total_heads']} 个头被选中 "
                      f"(总体计算量减少 {savings['computation_reduced']:.1f}%)")

                with open(train_log_file, 'a') as f:
                    f.write(
                        f"Epoch {epoch} Head Selection Summary: {savings['selected_heads']}/{savings['total_heads']} heads, "
                        f"Overall Reduction: {savings['computation_reduced']:.1f}%\n")

        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, avg_loss, epoch_time))
        with open(train_log_file, 'a') as f:
            f.write("Epoch {} , average loss {} , epoch_time {}\n".format(
                epoch, avg_loss, epoch_time))
        epoch_losses.append(avg_loss)

        # 保存模型
    save_model(model_gat, args.data, epoch, args.output_folder)


def train_conv(args):
    # 创建输出文件夹
    conv_output_folder = f"{args.output_folder}/conv/"
    if not os.path.exists(conv_output_folder):
        os.makedirs(conv_output_folder)

    #创建模型时使用相同的头选择参数
    print("Defining model")
    model_gat = GrCNet(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT,
                                use_head_selection=args.use_head_selection,  # 保持一致
                                head_selection_ratio=args.head_selection_ratio)  # 保持一致

    print("Only Conv model trained")
    model_conv = GrCNetConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()

    model_gat.load_state_dict(
        torch.load(f'{args.output_folder}/trained_{args.epochs_gat - 1}.pth'),
        strict=False
    )

    # 确保嵌入参数正确传递
    model_conv.final_entity_embeddings.data = model_gat.final_entity_embeddings.data
    model_conv.final_relation_embeddings.data = model_gat.final_relation_embeddings.data

    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()
    train_log_file = f"{conv_output_folder}/conv_train_log.txt"

    print("Starting Conv training from scratch")
    with open(train_log_file, "w") as f:
        f.write("Iteration,Iteration_time,Iteration_loss,Epoch,Average_loss,Epoch_time\n")
    start_epoch = 0

    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    for epoch in range(start_epoch, args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                                          len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()
            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))
            with open(train_log_file, 'a') as f:
                f.write("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}\n".format(
                    iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_time = time.time() - start_time
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, avg_loss, epoch_time))
        with open(train_log_file, 'a') as f:
            f.write("Epoch {} , average loss {} , epoch_time {}\n".format(
                epoch, avg_loss, epoch_time))
        epoch_losses.append(avg_loss)

        # 保存模型
    save_model(model_conv, args.data, epoch, conv_output_folder)


def evaluate_conv(args, unique_entities):
    model_conv = GrCNetConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    
    conv_checkpoint_path = f'{args.output_folder}conv/trained_{args.epochs_conv - 1}.pth'
    if os.path.exists(conv_checkpoint_path):
        model_conv.load_state_dict(torch.load(conv_checkpoint_path), strict=False)
    else:
        print(f"警告: 找不到卷积模型检查点 {conv_checkpoint_path}")
        return

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_conv, unique_entities)


print("自动检测训练状态...")
train_gat(args)
train_conv(args)
evaluate_conv(args, Corpus_.unique_entities_train)
