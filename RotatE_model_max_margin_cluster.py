#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import *
import json
import logging
import os
import gensim
import numpy as np
import torch
from kmeans import *


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#from model import KGEModel
def hinge_loss(positive_score, negative_score, gamma):
    err = positive_score - negative_score + gamma
    max_err = err.clamp(0)
    return max_err


class KGEModel(nn.Module):
    def __init__(self, params, model_name, dict_local, init, E_init, R_init, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.p = params
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.embed_loc = dict_local
        self.E_init = E_init
        self.R_init = R_init
        self.init = init

        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        ''' Intialize embeddings '''
        if self.init == 'crawl':
            self.entity_embedding = nn.Parameter(torch.from_numpy(self.E_init))
            self.relation_embedding = nn.Parameter(torch.from_numpy(self.R_init))
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        if model_name == 'pRotatE' or model_name == 'new_rotate':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'new_rotate']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single', iscluster=False, num_clusters=6000):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            if iscluster:
                entity_sample = torch.cat((sample[:, 0], sample[:, 2]), 0)
                entity_index = torch.unique(entity_sample)
                relation_index = torch.unique(sample[:, 1])

                entity = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=entity_index
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=relation_index
                ).unsqueeze(1)

            else:
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:, 0]
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            #print('head_part[:, 0]:', type(head_part[:, 0]), head_part[:, 0])
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            #print('head_part[:, 1]:', type(head_part[:, 1]), head_part[:, 1])
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            #print('tail_part.view(-1):', type(tail_part.view(-1)), tail_part.view(-1))
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'new_rotate': self.new_rotate
        }

        if iscluster:
            ent2embed = entity.squeeze(1)
            # rel2embed = relation.squeeze(1) # there is no use of rel2embed
            positive_score, negative_score = kmeans_train(dataset=ent2embed, num_centers=num_clusters)
            # loss = max_margin_loss(positive_score, negative_score, self.p.single_gamma)
            # loss = max_margin_loss(positive_score, negative_score)
            loss = positive_loss(positive_score)
            score = loss

        else:
            if self.model_name in model_func:
                score = model_func[self.model_name](head, relation, tail, mode)
            else:
                raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        # score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        score = torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        # score = self.gamma.item() - score.sum(dim=2)
        score = score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        # score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        score = score.sum(dim=2) * self.modulus
        return score

    def new_rotate(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)
        cos_relation = torch.cos(phase_relation)
        sin_relation = torch.sin(phase_relation)

        #head = head / (self.embedding_range.item() / pi)
        #tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            x_score = head * cos_relation - tail * cos_relation
            y_score = head * sin_relation - tail * sin_relation
        else:
            x_score = head * cos_relation - tail * cos_relation
            y_score = head * sin_relation - tail * sin_relation
            #x_score = tail * cos_relation - head * cos_relation
            #y_score = tail * sin_relation - head * sin_relation

        score = x_score * y_score
        # score = torch.abs(x_score + y_score)
        # score = self.gamma.item() - score.sum(dim=2) # self.modulus # 0.7944 30000epoch:0.21==0.7826
        # score = score.sum(dim=2) * self.modulus  # 0.785  30000epoch:0.20==00.7701
        # score = score.sum(dim=2) # 0.22==0.8029  30000epoch:0.29==0.7753
        # score = self.gamma.item() - torch.norm(score, p=1, dim=2)  # 0.35==0.8015  30000epoch:0.45==0.7777
        score = torch.norm(score, p=1, dim=2)  # 0.27==0.772  30000epoch:0.25==0.7874

        return score


    
    @staticmethod
    def train_step(args, model, optimizer, train_iterator):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        negative_sample_size = int(args.single_negative_sample_size)
        gamma = torch.full((1, negative_sample_size), float(args.single_gamma))  # 返回大小为sizes,单位值为fill_value的矩阵

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            gamma = gamma.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        #if args.negative_adversarial_sampling:
            ##In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            #negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              #* F.logsigmoid(-negative_score)).sum(dim=1)
        #else:
            #negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = positive_score.repeat(1, negative_sample_size)

        loss = hinge_loss(positive_score, negative_score, gamma)

        #positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            loss = loss.sum()
        else:
            loss = (subsampling_weight * loss).sum()/subsampling_weight.sum()
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3)**3 +
                model.relation_embedding.norm(p=3).norm(p=3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return log

    @staticmethod
    def cluster_triple_step(args, model, optimizer, train_iterator, num_clusters):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        negative_sample_size = int(args.single_negative_sample_size)
        # gamma = torch.full((1, negative_sample_size), float(args.single_gamma))  # 返回大小为sizes,单位值为fill_value的矩阵

        model.train()
        optimizer.zero_grad()
        positive_sample, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()

        loss = model(positive_sample, mode=mode, iscluster=True, num_clusters=num_clusters)

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()
        # print('cluster loss:', loss)

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return log

    @staticmethod
    def cluster_train_step(args, model, optimizer, entity_iterator):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        negative_sample_size = int(args.cross_negative_sample_size)
        gamma = torch.full((1, negative_sample_size), float(args.cross_gamma))  # 返回大小为sizes,单位值为fill_value的矩阵

        model.train()

        optimizer.zero_grad()

        # positive_sample, subsampling_weight = next(entity_iterator)
        positive_sample = next(entity_iterator)
        print('positive_sample:', type(positive_sample), positive_sample)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            # subsampling_weight = subsampling_weight.cuda()
            gamma = gamma.cuda()

        print('positive_sample:', type(positive_sample), positive_sample)
        positive_score = model(positive_sample)
        positive_score = positive_score.repeat(1, negative_sample_size)
        negative_score = positive_sample
        loss = hinge_loss(positive_score, negative_score, gamma)

        loss = loss.sum()
        exit()

        # if args.uni_weight:
        #     loss = loss.sum()
        # else:
        #     loss = (subsampling_weight * loss).sum() / subsampling_weight.sum()


        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()
        exit()

        log = {
            **regularization_log,
            'loss': loss.item()
        }

        return log


    def override_config(self):
        '''
        Override model and data configuration
        '''

        with open(os.path.join(self.p.init_checkpoint, 'config.json'), 'r') as fjson:
            argparse_dict = json.load(fjson)
        self.p.model = argparse_dict['model']
        self.p.double_entity_embedding = argparse_dict['double_entity_embedding']
        self.p.double_relation_embedding = argparse_dict['double_relation_embedding']
        self.p.hidden_dim = argparse_dict['hidden_dim']
        self.p.test_batch_size = argparse_dict['test_batch_size']

    def save_model(self, model, optimizer, save_variable_list):
        '''
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        '''

        argparse_dict = vars(self)
        with open(os.path.join(self.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(self.save_path, 'checkpoint')
        )

        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(self.save_path, 'entity_embedding'),
            entity_embedding
        )

        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(self.save_path, 'relation_embedding'),
            relation_embedding
        )

    def set_logger(self):
        '''
        Write logs to checkpoint and console
        '''

        if self.p.do_train:
            log_file = os.path.join(self.p.out_path or self.p.init_checkpoint, 'train.log')
        else:
            log_file = os.path.join(self.p.out_path or self.p.init_checkpoint, 'test.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def log_metrics(self, mode, step, metrics):
        '''
        Print the evaluation logs
        '''
        for metric in metrics:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))