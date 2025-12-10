import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import numpy as np

CUDA = torch.cuda.is_available()

class GrCNetSpmmFunction(torch.autograd.Function):
    """GrCNet稀疏矩阵乘法自定义函数"""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]
        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            if CUDA:
                edge_sources = edge_sources.cuda()
            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None

class GrCNetSpmm(nn.Module):
    """GrCNet稀疏矩阵乘法模块封装"""
    def forward(self, edge, edge_w, N, E, out_features):
        return GrCNetSpmmFunction.apply(edge, edge_w, N, E, out_features)

class GrCNetHeadSelector(nn.Module):
    """
    GrCNet基于粒度权重的多头注意力筛选器
    根据实体-关系粒度权重动态选择相关的注意力头
    """

    def __init__(self, num_heads, num_relations, num_entities, selection_ratio=0.5):
        super(GrCNetHeadSelector, self).__init__()
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.num_entities = num_entities
        self.selection_ratio = selection_ratio

        self.head_relation_affinity = nn.Parameter(
            torch.randn(num_heads, num_relations)
        )

        self.head_importance_weights = nn.Parameter(torch.ones(num_heads))

        self.granularity_fusion_alpha = nn.Parameter(torch.tensor(0.7))

    def forward(self, edge_list, edge_type, granularity_matrix, training=True):
        """
        基于粒度权重选择相关的注意力头

        Args:
            edge_list: 边列表 [2, E]
            edge_type: 边类型 [E]
            granularity_matrix: 粒度权重矩阵 Q [num_entities, num_relations]
            training: 是否训练模式

        Returns:
            head_mask: 注意力头掩码 [num_heads]
            head_scores: 头重要性分数 [num_heads]
        """
        batch_size = edge_list.shape[1] if edge_list.dim() > 1 else 1

        relation_based_scores = self._compute_relation_based_scores(edge_type)

        granularity_based_scores = self._compute_granularity_based_scores(
            edge_list, edge_type, granularity_matrix
        )

        fusion_alpha = torch.sigmoid(self.granularity_fusion_alpha)
        combined_scores = (
                fusion_alpha * granularity_based_scores +
                (1 - fusion_alpha) * relation_based_scores
        )

        final_scores = combined_scores * torch.sigmoid(self.head_importance_weights)

        if training:
            head_mask = self._soft_head_selection(final_scores)
        else:
            head_mask = self._hard_head_selection(final_scores)

        return head_mask, final_scores.detach()
    
    def _compute_relation_based_scores(self, edge_type):
        """计算基于关系的头重要性分数"""
        if edge_type.numel() == 0:
            return torch.ones(self.num_heads, device=edge_type.device) / self.num_heads
        
        safe_edge_type = torch.clamp(edge_type, 0, self.num_relations - 1)
        
        unique_relations, counts = torch.unique(safe_edge_type, return_counts=True)
        relation_weights = torch.zeros(self.num_relations, device=edge_type.device)
        
        valid_mask = (unique_relations >= 0) & (unique_relations < self.num_relations)
        unique_relations = unique_relations[valid_mask]
        counts = counts[valid_mask]
        
        if len(unique_relations) > 0:
            relation_weights[unique_relations] = counts.float() / counts.sum()
        else:
            relation_weights = torch.ones_like(relation_weights) / self.num_relations

        head_scores = torch.matmul(self.head_relation_affinity, relation_weights)
        return torch.softmax(head_scores, dim=0)

    def _compute_granularity_based_scores(self, edge_list, edge_type, granularity_matrix):
        """计算基于粒度的头重要性分数"""
        if edge_list.dim() == 2:
            source_nodes = edge_list[0]
            target_nodes = edge_list[1]
        else:
            source_nodes = edge_list
            target_nodes = edge_list

        source_granularity = granularity_matrix[source_nodes, edge_type]
        target_granularity = granularity_matrix[target_nodes, edge_type]

        edge_granularity = (source_granularity + target_granularity) / 2

        granularity_mean = edge_granularity.mean()
        granularity_std = edge_granularity.std()

        base_scores = torch.linspace(0, 1, self.num_heads, device=edge_type.device)
        granularity_factor = torch.sigmoid((granularity_mean - 0.5) * 10)

        head_scores = base_scores * granularity_factor + (1 - base_scores) * (1 - granularity_factor)

        return torch.softmax(head_scores, dim=0)

    def _soft_head_selection(self, head_scores):
        """软选择：使用Gumbel-Softmax进行可微分的头选择"""
        temperature = 0.1
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(head_scores) + 1e-10) + 1e-10)
        noisy_scores = (head_scores + gumbel_noise) / temperature

        selection_probs = torch.sigmoid(noisy_scores)

        k = int(self.num_heads * self.selection_ratio)
        if k < self.num_heads:
            _, topk_indices = torch.topk(selection_probs, k)
            mask = torch.zeros_like(selection_probs)
            mask[topk_indices] = 1.0
            return mask
        else:
            return selection_probs

    def _hard_head_selection(self, head_scores):
        """硬选择：选择top-k个注意力头"""
        k = int(self.num_heads * self.selection_ratio)
        _, topk_indices = torch.topk(head_scores, k)
        mask = torch.zeros(self.num_heads, device=head_scores.device)
        mask[topk_indices] = 1.0
        return mask

class GrCNetConvKB(nn.Module):
    """GrCNet卷积知识库嵌入模型"""
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, (1, input_seq_len))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)
        
        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        
        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output

class GrCNetAttentionLayer(nn.Module):
    """
    GrCNet稀疏图注意力层 - 集成多头注意力筛选
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True, 
                 use_head_selection=False, num_relations=0, head_selection_ratio=0.5):
        super(GrCNetAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim
        self.use_head_selection = use_head_selection

        self.a = nn.Parameter(torch.zeros(size=(out_features, 2 * in_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        if use_head_selection:
            self.head_selector = GrCNetHeadSelector(
                num_heads=1,
                num_relations=num_relations,
                num_entities=num_nodes,
                selection_ratio=head_selection_ratio
            )

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = GrCNetSpmm()

        self.attention_weights = None
        self.raw_attention_before_granularity = None
        self.granularity_scores = None
        self.head_mask = None

    def forward(self, input, edge, edge_embed, edge_type, granularity_labels):
        N = input.size()[0]

        if self.use_head_selection:
            self.head_mask, head_scores = self.head_selector(
                edge, edge_type, granularity_labels, self.training
            )
            if torch.sum(self.head_mask) == 0:
                self.head_mask = torch.ones_like(self.head_mask)
        else:
            self.head_mask = torch.ones(1, device=input.device)

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :]), dim=1).t()

        edge_m = self.a.mm(edge_h)

        self.raw_attention_before_granularity = edge_m.detach().cpu().numpy()
        
        row_idx = edge[0, :]
        col_idx = edge[1, :]

        gran_scores_i = granularity_labels[row_idx, edge_type]
        gran_scores_j = granularity_labels[col_idx, edge_type]
        gran_scores = (gran_scores_i + gran_scores_j)/2
        self.granularity_scores = gran_scores.detach().cpu().numpy()

        if self.use_head_selection:
            head_mask_expanded = self.head_mask.view(-1, 1).expand_as(edge_m)
            gran_scores = gran_scores.unsqueeze(0) * head_mask_expanded
        else:
            gran_scores = gran_scores.unsqueeze(0)
        
        weighted_edge_m = edge_m * gran_scores
        
        raw_e = self.a_2.mm(weighted_edge_m)
        activated = -self.leakyrelu(raw_e)
        powers = activated.squeeze(0)

        edge_e = torch.exp(powers).unsqueeze(1)
        self.attention_weights = edge_e.detach().cpu().numpy().flatten()

        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm_final(edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        edge_e = edge_e.squeeze(1)
        edge_e = self.dropout(edge_e)

        edge_w = (edge_e * weighted_edge_m).t()

        h_prime = self.special_spmm_final(edge, edge_w, N, edge_w.shape[0], self.out_features)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime), self.attention_weights
        else:
            return h_prime, self.attention_weights

    def get_attention_analysis_data(self):
        """获取注意力分析数据 - 包含头筛选信息"""
        if self.attention_weights is None:
            return None

        analysis_data = {
            'final_attention_weights': self.attention_weights,
            'raw_attention_before_granularity': self.raw_attention_before_granularity,
            'granularity_scores': self.granularity_scores,
            'head_mask': self.head_mask.detach().cpu().numpy() if self.head_mask is not None else None,
            'use_head_selection': self.use_head_selection
        }

        return analysis_data

    def reset_attention_data(self):
        """重置注意力数据"""
        self.attention_weights = None
        self.raw_attention_before_granularity = None
        self.granularity_scores = None
        self.head_mask = None

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GrCNetGAT(nn.Module):
    """
    GrCNet改进的稀疏图注意力网络 - 集成多头注意力筛选
    """

    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads, 
                 use_head_selection=False, head_selection_ratio=1):
        super(GrCNetGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.use_head_selection = use_head_selection
        self.nheads = nheads

        self.attentions = [GrCNetAttentionLayer(num_nodes, nfeat, nhid, relation_dim,
                                                dropout=dropout, alpha=alpha, concat=True,
                                                use_head_selection=use_head_selection,
                                                num_relations=relation_dim,
                                                head_selection_ratio=head_selection_ratio)
                          for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = GrCNetAttentionLayer(num_nodes, nhid * nheads, nheads * nhid, nheads * nhid,
                                            dropout=dropout, alpha=alpha, concat=False,
                                            use_head_selection=use_head_selection,
                                            num_relations=relation_dim,
                                            head_selection_ratio=head_selection_ratio)

        if use_head_selection:
            self.multi_head_selector = GrCNetHeadSelector(
                num_heads=nheads,
                num_relations=relation_dim,
                num_entities=num_nodes,
                selection_ratio=head_selection_ratio
            )

        self.attention_weights_all_heads = None
        self.final_attention_weights = None
        self.head_selection_mask = None

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, granularity_labels):
        N = entity_embeddings.size()[0]

        if self.use_head_selection:
            self.head_selection_mask, head_scores = self.multi_head_selector(
                edge_list, edge_type, granularity_labels, self.training
            )
        else:
            self.head_selection_mask = torch.ones(self.nheads, device=entity_embeddings.device)

        attention_outputs = []
        self.attention_weights_all_heads = []

        for i, att in enumerate(self.attentions):
            if self.use_head_selection and self.head_selection_mask[i] < 0.5:
                zero_output = torch.zeros(N, att.out_features, device=entity_embeddings.device)
                attention_outputs.append(zero_output)
                self.attention_weights_all_heads.append(np.zeros(edge_list.shape[1]))
                continue

            att.reset_attention_data()
            output, att_weights = att(entity_embeddings, edge_list, edge_embed, 
                                     edge_type, granularity_labels)
            attention_outputs.append(output)
            self.attention_weights_all_heads.append(att_weights)

        x = torch.cat(attention_outputs, dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]

        self.out_att.reset_attention_data()
        x, final_att_weights = self.out_att(x, edge_list, edge_embed, 
                                           edge_type, granularity_labels)

        x = F.elu(x)
        self.final_attention_weights = final_att_weights

        attention_weights_dict = {
            'multihead_attention_weights': self.attention_weights_all_heads,
            'final_attention_weights': self.final_attention_weights,
            'head_selection_mask': self.head_selection_mask.detach().cpu().numpy() if self.head_selection_mask is not None else None,
            'num_heads': self.nheads,
            'num_selected_heads': torch.sum(self.head_selection_mask).item() if self.head_selection_mask is not None else self.nheads,
            'edge_indices': edge_list.detach().cpu().numpy() if edge_list is not None else None,
            'edge_types': edge_type.detach().cpu().numpy() if edge_type is not None else None
        }

        return x, out_relation_1, attention_weights_dict

    def get_attention_analysis_data(self):
        """获取注意力分析数据 - 包含头筛选信息"""
        if self.attention_weights_all_heads is None:
            return None

        analysis_data = {
            'multihead_attention_weights': self.attention_weights_all_heads,
            'final_attention_weights': self.final_attention_weights,
            'head_selection_mask': self.head_selection_mask.detach().cpu().numpy() if self.head_selection_mask is not None else None,
            'num_heads': self.nheads,
            'num_selected_heads': torch.sum(self.head_selection_mask).item() if self.head_selection_mask is not None else self.nheads,
            'use_head_selection': self.use_head_selection
        }

        head_analysis_data = []
        for i, att in enumerate(self.attentions):
            head_data = att.get_attention_analysis_data()
            if head_data:
                head_data['head_index'] = i
                head_data['head_selected'] = self.head_selection_mask[i].item() > 0.5 if self.head_selection_mask is not None else True
                head_analysis_data.append(head_data)

        analysis_data['per_head_analysis'] = head_analysis_data

        output_head_data = self.out_att.get_attention_analysis_data()
        if output_head_data:
            output_head_data['head_index'] = 'output'
            analysis_data['output_head_analysis'] = output_head_data

        return analysis_data

class GrCNet(nn.Module):
    """
    GrCNet知识图谱GAT模型 - 支持多头注意力筛选
    """

    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, use_head_selection=False, head_selection_ratio=0.5):
        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha
        self.use_head_selection = use_head_selection
        self.head_selection_ratio = head_selection_ratio

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = GrCNetGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                     self.drop_GAT, self.alpha, self.nheads_GAT_1,
                                     use_head_selection=use_head_selection,
                                     head_selection_ratio=head_selection_ratio)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

        self.attention_weights = None
        self.computation_savings = None

    def forward(self, Corpus_, adj, batch_inputs):
        edge_list = adj[0]
        edge_type = adj[1]

        if CUDA:
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        granularity_matrix = Corpus_.granularity_matrix

        self.entity_embeddings.data = F.normalize(self.entity_embeddings.data, p=2, dim=1).detach()

        out_entity_1, out_relation_1, attention_weights = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, granularity_labels=granularity_matrix)

        if self.use_head_selection and 'num_selected_heads' in attention_weights:
            total_heads = attention_weights['num_heads']
            selected_heads = attention_weights['num_selected_heads']
            savings_ratio = 1 - (selected_heads / total_heads)
            self.computation_savings = {
                'total_heads': total_heads,
                'selected_heads': selected_heads,
                'savings_ratio': savings_ratio,
                'computation_reduced': savings_ratio * 100
            }

        self.attention_weights = attention_weights

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1, attention_weights

    def get_attention_analysis_data(self):
        """获取注意力分析数据 - 包含头筛选信息"""
        analysis_data = {}
        if hasattr(self, 'attention_weights') and self.attention_weights:
            analysis_data.update(self.attention_weights)
        
        if hasattr(self, 'computation_savings') and self.computation_savings:
            analysis_data['computation_savings'] = self.computation_savings
            
        return analysis_data if analysis_data else None

class GrCNetConvOnly(nn.Module):
    """
    GrCNet仅卷积的知识图谱模型
    使用预训练的GAT嵌入，只训练卷积部分
    """

    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        """
        初始化仅卷积模型

        Args:
            initial_entity_emb: 初始实体嵌入
            initial_relation_emb: 初始关系嵌入
            entity_out_dim: 实体输出维度列表
            relation_out_dim: 关系输出维度列表
            drop_GAT: GAT Dropout概率
            drop_conv: 卷积Dropout概率
            alpha: GAT LeakyReLU参数
            alpha_conv: 卷积LeakyReLU参数
            nheads_GAT: 多头注意力头数列表
            conv_out_channels: 卷积输出通道数
        """
        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = GrCNetConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        """
        前向传播

        Args:
            Corpus_: 数据语料库
            adj: 邻接矩阵
            batch_inputs: 批次输入数据

        Returns:
            卷积输出得分
        """
        conv_input = torch.cat((
            self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1),
            self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1),
            self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)
        ), dim=1)

        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        """
        批次测试方法

        Args:
            batch_inputs: 测试批次输入

        Returns:
            测试得分
        """
        conv_input = torch.cat((
            self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1),
            self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1),
            self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)
        ), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv