import torch
import os
import numpy as np
import time
import queue
import random
import pickle
from copy import deepcopy

CUDA = torch.cuda.is_available()


def read_entity_from_id(filename='./data/WN18RR/entity2id.txt'):
    entity2id = {}
    print(filename)
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)


def build_data(path='./data/WN18RR/', is_unweigted=False, directed=True):

    entity2id = read_entity_from_id(path + '/entity2id.txt')
    relation2id = read_relation_from_id(path + '/relation2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    # print(train_triples)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return train_triples,(train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train


def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print(folder_name)
    print("Done saving Model")


class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, entity2id,
                 relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train,
                 granularity_matrix=None):
        self.args = args
        self.cache_dir = os.path.join(args.data, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.train_triples = train_data[0]

        # Converting to sparse tensor
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(
            len(self.valid_triples_dict), len(self.train_indices), len(self.validation_indices),
            len(self.test_indices)))
        if granularity_matrix is not None:
            # 把 numpy Q 转成 torch.Tensor
            Q_tensor = torch.from_numpy(granularity_matrix)  # [N, N]
            if CUDA:
                Q_tensor = Q_tensor.cuda()
            # 保持为一个属性
            self.granularity_matrix = Q_tensor
        else:
            self.granularity_matrix = None
        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

    def get_iteration_batch(self, iter_num):
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))

            self.batch_indices[:self.batch_size,
            :] = self.train_indices[indices, :]
            self.batch_values[:self.batch_size,
            :] = self.train_values[indices, :]

            last_index = self.batch_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                        0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                                        (self.invalid_valid_ratio // 2) + \
                                        (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0],
                               self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                        2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

        else:
            last_iter_size = len(self.train_indices) - \
                             self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))
            self.batch_indices[:last_iter_size,
            :] = self.train_indices[indices, :]
            self.batch_values[:last_iter_size,
            :] = self.train_values[indices, :]

            last_index = last_iter_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                        0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                                        (self.invalid_valid_ratio // 2) + \
                                        (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0],
                               self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                        2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x

    def get_validation_pred(self, args, model, unique_entities):
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            entity_list = [j for i, j in self.entity2id.items()]

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_indices.shape[0]):
                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))
                new_x_batch_tail = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))

                if (batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys():
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_dict.keys():
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)

                import math
                # Have to do this, because it doesn't fit in memory

                if 'WN' in args.data:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_head.shape[0] / 4))

                    scores1_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[:num_triples_each_shot, :]).cuda())
                    scores2_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_head = model.batch_test(torch.LongTensor(
                        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())

                    scores_head = torch.cat(
                        [scores1_head, scores2_head, scores3_head, scores4_head], dim=0)
                else:
                    scores_head = model.batch_test(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here

                if 'WN' in args.data:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_tail.shape[0] / 4))

                    scores1_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                    scores2_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_tail = model.batch_test(torch.LongTensor(
                        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())

                    scores_tail = torch.cat(
                        [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)

                else:
                    scores_tail = model.batch_test(new_x_batch_tail)

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                print("sample - ", ranks_head[-1], ranks_tail[-1])

            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            eval_log_file = f"{args.output_folder}/conv_eval_log.txt"
            with open(eval_log_file, "w") as f:
                f.write("hits,mean\n")  # 写入表头
            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))
            with open(eval_log_file, 'a') as f:
                f.write("here {}".format(len(ranks_head)) + "\nCurrent iteration time {}".format(
                    time.time() - start_time) + "Stats for replacing head are -> "
                        + "Current iteration Hits@100 are {}".format(
                    hits_at_100_head / float(len(ranks_head))) + '' + "Current iteration Hits@10 are {}".format(
                    hits_at_ten_head / len(ranks_head)) + '' + "Current iteration Hits@3 are {}".format(
                    hits_at_three_head / len(ranks_head)) + '' + "Current iteration Hits@1 are {}".format(
                    hits_at_one_head / len(ranks_head)) + '' + "Current iteration Mean rank {}".format(
                    sum(ranks_head) / len(ranks_head)) + '' + "Current iteration Mean Reciprocal Rank {}".format(
                    sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)) + '\n')
            print("\nStats for replacing tail are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_tail / len(ranks_head)))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_tail / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_tail / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_tail / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_tail) / len(ranks_tail)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))
            with open(eval_log_file, 'a') as f:
                f.write("\nStats for replacing tail are ->"
                        + "Current iteration Hits@100 are {}".format(
                    hits_at_100_tail / len(ranks_head)) + '' + "Current iteration Hits@10 are {}".format(
                    hits_at_ten_tail / len(ranks_head)) + '' + "Current iteration Hits@3 are {}".format(
                    hits_at_three_tail / len(ranks_head)) + '' + "Current iteration Hits@1 are {}".format(
                    hits_at_one_tail / len(ranks_head)) + '' + "Current iteration Mean rank {}".format(
                    sum(ranks_tail) / len(ranks_tail)) + '' + "Current iteration Mean Reciprocal Rank {}".format(
                    sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)) + '\n')
            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))
        with open(eval_log_file, 'a') as f:
            f.write("\nAveraged stats for replacing head are ->"
                    + "Hits@100 are {}".format(
                sum(average_hits_at_100_head) / len(average_hits_at_100_head)) + '' + "Hits@10 are {}".format(
                sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)) + '' + "Hits@3 are {}".format(
                sum(average_hits_at_three_head) / len(
                    average_hits_at_three_head)) + '' + "Current iteration Hits@1 are {}".format(
                hits_at_one_tail / len(ranks_head)) + '' + "Mean rank {}".format(
                sum(average_mean_rank_head) / len(average_mean_rank_head)) + '' + "Mean Reciprocal Rank {}".format(
                sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)) + '\n')
        print("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))
        with open(eval_log_file, 'a') as f:
            f.write("\nAveraged stats for replacing tail are -> "
                    + "Hits@100 are {}".format(
                sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) + '' + "Hits@10 are {}".format(
                sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) + '' + "Hits@3 are {}".format(
                sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) + '' + "Hits@1 are {}".format(
                sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) + '' + "Mean rank {}".format(
                sum(average_mean_rank_tail) / len(average_mean_rank_tail)) + '' + "Mean Reciprocal Rank {}".format(
                sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) + '\n')
        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))
        with open(eval_log_file, 'a') as f:
            f.write("\nCumulative stats are ->  "
                    + "Hits@100 are {}".format(cumulative_hits_100) + '' + "Hits@10 are {}".format(
                cumulative_hits_ten) + '' + "Hits@3 are {}".format(cumulative_hits_three) +
                    '' + "Hits@1 are {}".format(cumulative_hits_one) + '' + "Mean rank {}".format(
                cumulative_mean_rank) + '' + "Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank) + '\n')

        # 保存验证结果
        results = {
            'average_hits_at_100_head': average_hits_at_100_head,
            'average_hits_at_100_tail': average_hits_at_100_tail,
            'average_hits_at_ten_head': average_hits_at_ten_head,
            'average_hits_at_ten_tail': average_hits_at_ten_tail,
            'average_hits_at_three_head': average_hits_at_three_head,
            'average_hits_at_three_tail': average_hits_at_three_tail,
            'average_hits_at_one_head': average_hits_at_one_head,
            'average_hits_at_one_tail': average_hits_at_one_tail,
            'average_mean_rank_head': average_mean_rank_head,
            'average_mean_rank_tail': average_mean_rank_tail,
            'average_mean_recip_rank_head': average_mean_recip_rank_head,
            'average_mean_recip_rank_tail': average_mean_recip_rank_tail,
            'cumulative_hits_100': cumulative_hits_100,
            'cumulative_hits_ten': cumulative_hits_ten,
            'cumulative_hits_three': cumulative_hits_three,
            'cumulative_hits_one': cumulative_hits_one,
            'cumulative_mean_rank': cumulative_mean_rank,
            'cumulative_mean_recip_rank': cumulative_mean_recip_rank
        }

        return results

    def get_validation_pred_gat(self, args, entity_embed, relation_embed, unique_entities):
        """GAT专用验证逻辑（使用验证集进行评估）"""
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()
            
            # 🆕 修改：使用验证集而不是测试集
            indices = [i for i in range(len(self.validation_indices))]
            batch_indices = self.validation_indices[indices, :]
            print(f"验证集长度: {len(self.validation_indices)}")
            entity_list = list(self.entity2id.values())  # 所有实体ID列表

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_indices.shape[0]):
                if i % 100 == 0:
                    print(f"已验证 {i}/{batch_indices.shape[0]} 个三元组")

                # 获取当前三元组的ID
                h_id, r_id, t_id = batch_indices[i]
                if h_id not in unique_entities or t_id not in unique_entities:
                    continue

                # -------------------------- 替换头实体验证（h, r, t）-> (?, r, t) --------------------------
                # 生成所有可能的头实体替换三元组
                new_x_batch_head = np.tile(batch_indices[i], (len(entity_list), 1))
                new_x_batch_head[:, 0] = entity_list  # 替换头实体为所有实体

                # 🆕 修改：过滤已存在的有效三元组（避免假阴性）- 使用所有有效三元组
                invalid_indices = []
                for idx in range(len(new_x_batch_head)):
                    triple = (new_x_batch_head[idx][0], new_x_batch_head[idx][1], new_x_batch_head[idx][2])
                    if triple in self.valid_triples_dict:
                        invalid_indices.append(idx)
                new_x_batch_head = np.delete(new_x_batch_head, invalid_indices, axis=0)

                # 插入原始正确三元组（作为参考）
                original_idx = len(new_x_batch_head)
                new_x_batch_head = np.insert(new_x_batch_head, 0, batch_indices[i], axis=0)

                # 计算GAT模型的得分（使用实体嵌入）
                # 得分公式：||h_embed + r_embed - t_embed||（与GAT训练时一致）
                h_embeds = entity_embed[torch.LongTensor(new_x_batch_head[:, 0]).cuda()]
                r_embeds = relation_embed[torch.LongTensor(new_x_batch_head[:, 1]).cuda()]
                t_embeds = entity_embed[torch.LongTensor(new_x_batch_head[:, 2]).cuda()]
                scores_head = torch.norm(h_embeds + r_embeds - t_embeds, p=1, dim=1)  # L1范数作为得分

                # 排序并计算排名（原始三元组在插入的第0位）
                sorted_scores, sorted_indices = torch.sort(scores_head, dim=0, descending=False)  # 得分越小越好
                rank = torch.where(sorted_indices == 0)[0].item() + 1  # +1是因为排名从1开始
                ranks_head.append(rank)
                reciprocal_ranks_head.append(1.0 / rank)

                # -------------------------- 替换尾实体验证（h, r, t）-> (h, r, ?) --------------------------
                new_x_batch_tail = np.tile(batch_indices[i], (len(entity_list), 1))
                new_x_batch_tail[:, 2] = entity_list  # 替换尾实体为所有实体

                # 🆕 修改：过滤已存在的有效三元组
                invalid_indices = []
                for idx in range(len(new_x_batch_tail)):
                    triple = (new_x_batch_tail[idx][0], new_x_batch_tail[idx][1], new_x_batch_tail[idx][2])
                    if triple in self.valid_triples_dict:
                        invalid_indices.append(idx)
                new_x_batch_tail = np.delete(new_x_batch_tail, invalid_indices, axis=0)

                # 插入原始正确三元组
                new_x_batch_tail = np.insert(new_x_batch_tail, 0, batch_indices[i], axis=0)

                # 计算得分
                h_embeds = entity_embed[torch.LongTensor(new_x_batch_tail[:, 0]).cuda()]
                r_embeds = relation_embed[torch.LongTensor(new_x_batch_tail[:, 1]).cuda()]
                t_embeds = entity_embed[torch.LongTensor(new_x_batch_tail[:, 2]).cuda()]
                scores_tail = torch.norm(h_embeds + r_embeds - t_embeds, p=1, dim=1)

                # 排序并计算排名
                sorted_scores, sorted_indices = torch.sort(scores_tail, dim=0, descending=False)
                rank = torch.where(sorted_indices == 0)[0].item() + 1
                ranks_tail.append(rank)
                reciprocal_ranks_tail.append(1.0 / rank)

            # 计算评估指标
            for rank in ranks_head:
                if rank <= 100:
                    hits_at_100_head += 1
                if rank <= 10:
                    hits_at_ten_head += 1
                if rank <= 3:
                    hits_at_three_head += 1
                if rank == 1:
                    hits_at_one_head += 1

            for rank in ranks_tail:
                if rank <= 100:
                    hits_at_100_tail += 1
                if rank <= 10:
                    hits_at_ten_tail += 1
                if rank <= 3:
                    hits_at_three_tail += 1
                if rank == 1:
                    hits_at_one_tail += 1

            # 输出当前迭代结果
            print(f"\n替换头实体的统计结果 -> ")
            print(f"Hits@100: {hits_at_100_head / len(ranks_head):.4f}")
            print(f"Hits@10: {hits_at_ten_head / len(ranks_head):.4f}")
            print(f"Hits@3: {hits_at_three_head / len(ranks_head):.4f}")
            print(f"Hits@1: {hits_at_one_head / len(ranks_head):.4f}")
            print(f"Mean Rank: {sum(ranks_head) / len(ranks_head):.4f}")
            print(f"Mean Reciprocal Rank: {sum(reciprocal_ranks_head) / len(reciprocal_ranks_head):.4f}")

            print(f"\n替换尾实体的统计结果 -> ")
            print(f"Hits@100: {hits_at_100_tail / len(ranks_tail):.4f}")
            print(f"Hits@10: {hits_at_ten_tail / len(ranks_tail):.4f}")
            print(f"Hits@3: {hits_at_three_tail / len(ranks_tail):.4f}")
            print(f"Hits@1: {hits_at_one_tail / len(ranks_tail):.4f}")
            print(f"Mean Rank: {sum(ranks_tail) / len(ranks_tail):.4f}")
            print(f"Mean Reciprocal Rank: {sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail):.4f}")

            # 累计平均结果
            average_hits_at_100_head.append(hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(hits_at_100_tail / len(ranks_tail))
            average_hits_at_ten_tail.append(hits_at_ten_tail / len(ranks_tail))
            average_hits_at_three_tail.append(hits_at_three_tail / len(ranks_tail))
            average_hits_at_one_tail.append(hits_at_one_tail / len(ranks_tail))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        # 输出最终平均结果
        print(f"\n验证集最终平均统计结果 -> ")
        print(
            f"平均Hits@100: {(sum(average_hits_at_100_head) + sum(average_hits_at_100_tail)) / (2 * len(average_hits_at_100_head)):.4f}")
        print(
            f"平均Hits@10: {(sum(average_hits_at_ten_head) + sum(average_hits_at_ten_tail)) / (2 * len(average_hits_at_ten_head)):.4f}")
        print(
            f"平均Hits@3: {(sum(average_hits_at_three_head) + sum(average_hits_at_three_tail)) / (2 * len(average_hits_at_three_head)):.4f}")
        print(
            f"平均Hits@1: {(sum(average_hits_at_one_head) + sum(average_hits_at_one_tail)) / (2 * len(average_hits_at_one_head)):.4f}")
        print(
            f"平均Mean Rank: {(sum(average_mean_rank_head) + sum(average_mean_rank_tail)) / (2 * len(average_mean_rank_head)):.4f}")
        print(
            f"平均MRR: {(sum(average_mean_recip_rank_head) + sum(average_mean_recip_rank_tail)) / (2 * len(average_mean_recip_rank_head)):.4f}")
        
        return {
            'average_mrr': (sum(average_mean_recip_rank_head) + sum(average_mean_recip_rank_tail)) / (2 * len(average_mean_recip_rank_head)),
            'average_hits_at_1': (sum(average_hits_at_one_head) + sum(average_hits_at_one_tail)) / (2 * len(average_hits_at_one_head)),
            'average_hits_at_3': (sum(average_hits_at_three_head) + sum(average_hits_at_three_tail)) / (2 * len(average_hits_at_three_head)),
            'average_hits_at_10': (sum(average_hits_at_ten_head) + sum(average_hits_at_ten_tail)) / (2 * len(average_hits_at_ten_head)),
            'average_hits_at_100': (sum(average_hits_at_100_head) + sum(average_hits_at_100_tail)) / (2 * len(average_hits_at_100_head)),
            'average_mean_rank': (sum(average_mean_rank_head) + sum(average_mean_rank_tail)) / (2 * len(average_mean_rank_head))
        }