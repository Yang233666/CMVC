import gensim
from helper import *
from skge.transe_soft import TransE_soft

import logging
import os

from torch.utils.data import DataLoader

#from dataloader import TrainDataset
#from dataloader import BidirectionalOneShotIterator
# from RotatE_dataloader import *
from RotatE_dataloader_max_margin import *
from RotatE_model_max_margin import KGEModel


class Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, logger):
        self.p = params
        self.logger = logger

        self.side_info = side_info
        self.ent2embed = {}  # Stores final embeddings learned for noun phrases
        self.rel2embed = {}  # Stores final embeddings learned for relation phrases
        self.ent2dict_embed = {}

    def fit(self):
        clean_ent_list, clean_rel_list = [], []
        if self.p.use_Entity_linking_dict:
            for ent in self.side_info.new_ent_list: clean_ent_list.append(ent.split('|')[0])
            for rel in self.side_info.new_rel_list: clean_rel_list.append(rel.split('|')[0])
        else:
            for ent in self.side_info.ent_list: clean_ent_list.append(ent.split('|')[0])
            for rel in self.side_info.rel_list: clean_rel_list.append(rel.split('|')[0])

        ''' Intialize embeddings '''
        if self.p.embed_init == 'crawl':
            fname1, fname2 = '1E_init', '1R_init'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate init word2vec dict embeddings')

                model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
                self.E_init = getEmbeddings(model, clean_ent_list, self.p.embed_dims)
                self.R_init = getEmbeddings(model, clean_rel_list, self.p.embed_dims)
                #self.R_init = getEmbeddings(model, self.side_info.rel_list, self.p.embed_dims)

                pickle.dump(self.E_init, open(fname1, 'wb'))
                pickle.dump(self.R_init, open(fname2, 'wb'))
            else:
                print('load init embeddings')
                self.E_init = pickle.load(open(fname1, 'rb'))
                self.R_init = pickle.load(open(fname2, 'rb'))

        else:
            print('generate init random embeddings')
            self.E_init = np.random.rand(len(clean_ent_list), self.p.embed_dims)
            self.R_init = np.random.rand(len(clean_rel_list), self.p.embed_dims)

        if self.p.use_Embedding_model:
            if self.p.init_checkpoint:
                KGEModel.override_config(self)

            # Write logs to checkpoint and console
            KGEModel.set_logger(self)

            # nentity, nrelation = len(self.side_info.new_ent_list), len(self.side_info.new_rel_list)
            nentity, nrelation = len(self.side_info.new_ent_list), len(self.side_info.new_rel_list)

            self.nentity = nentity
            self.nrelation = nrelation

            logging.info('Model: %s' % self.p.model)
            logging.info('#entity: %d' % nentity)
            logging.info('#relation: %d' % nrelation)

            train_triples = self.side_info.new_trpIds
            train_sim = [1] * len(train_triples)
            seed_triples = self.side_info.seed_trpIds
            print('seed_triples:', len(seed_triples))
            seed_sim = train_sim + self.side_info.seed_sim
            #seed_triples = train_triples + seed_triples
            #train_triples = train_triples + seed_triples

            logging.info('#train: %d' % len(train_triples))
            logging.info('#seed: %d' % len(seed_triples))

            kge_model = KGEModel(
                model_name=self.p.model,
                dict_local=self.p.embed_loc,
                init=self.p.embed_init,
                E_init=self.E_init,
                R_init=self.R_init,
                nentity=nentity,
                nrelation=nrelation,
                hidden_dim=self.p.hidden_dim,
                gamma=self.p.gamma,
                double_entity_embedding=self.p.double_entity_embedding,
                double_relation_embedding=self.p.double_relation_embedding
            )

            logging.info('Model Parameter Configuration:')
            for name, param in kge_model.named_parameters():
                logging.info(
                    'Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

            if self.p.cuda:
                kge_model = kge_model.cuda()

            if self.p.do_train:
                # Set training dataloader iterator
                train_dataloader_head = DataLoader(
                    TrainDataset(train_triples, nentity, nrelation, self.p.negative_sample_size, 'head-batch'),
                    batch_size=self.p.batch_size,
                    shuffle=True,
                    num_workers=max(1, self.p.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )

                train_dataloader_tail = DataLoader(
                    TrainDataset(train_triples, nentity, nrelation, self.p.negative_sample_size, 'tail-batch'),
                    batch_size=self.p.batch_size,
                    shuffle=True,
                    num_workers=max(1, self.p.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )

                train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

                # Set training configuration
                current_learning_rate = self.p.learning_rate
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                if self.p.warm_up_steps:
                    warm_up_steps = self.p.warm_up_steps
                else:
                    warm_up_steps = self.p.max_steps // 2

            if self.p.init_checkpoint:
                # Restore model from checkpoint directory
                logging.info('Loading checkpoint %s...' % self.p.init_checkpoint)
                checkpoint = torch.load(os.path.join(self.p.init_checkpoint, 'checkpoint'))
                init_step = checkpoint['step']
                kge_model.load_state_dict(checkpoint['model_state_dict'])
                if self.p.do_train:
                    current_learning_rate = checkpoint['current_learning_rate']
                    warm_up_steps = checkpoint['warm_up_steps']
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logging.info('Ramdomly Initializing %s Model...' % self.p.model)
                init_step = 0

            step = init_step

            logging.info('Start Training...')
            logging.info('init_step = %d' % init_step)
            logging.info('batch_size = %d' % self.p.batch_size)
            logging.info('negative_adversarial_sampling = %d' % self.p.negative_sample_size)
            logging.info('hidden_dim = %d' % self.p.hidden_dim)
            logging.info('gamma = %f' % self.p.gamma)
            logging.info('negative_adversarial_sampling = %s' % str(self.p.negative_adversarial_sampling))
            if self.p.negative_adversarial_sampling:
                logging.info('adversarial_temperature = %f' % self.p.adversarial_temperature)

            # Set valid dataloader as it would be evaluated during training

            if self.p.do_train:
                logging.info('learning_rate = %d' % current_learning_rate)

                training_logs = []
                if self.p.use_cross_seed:

                    # use this when do not update seeds
                    seed_dataloader = DataLoader(
                        SeedDataset(seed_triples, nentity, nrelation, seed_sim),
                        batch_size=self.p.batch_size,
                        shuffle=True,
                        num_workers=max(1, self.p.cpu_num // 2),
                        collate_fn=SeedDataset.collate_fn
                    )
                    seed_iterator = SeeddirectionalOneShotIterator(seed_dataloader)

                # Training Loop
                for step in range(init_step, self.p.max_steps):

                    log = kge_model.train_step(self.p, kge_model, optimizer, train_iterator)
                    training_logs.append(log)

                    if not self.p.use_cross_seed:
                        if step % self.p.update_seed_steps == 0:
                            '''use this when update seeds 
                            seed_dataloader = DataLoader(
                                SeedDataset(seed_triples, nentity, nrelation, seed_sim),
                                batch_size=self.p.batch_size,
                                shuffle=True,
                                num_workers=max(1, self.p.cpu_num // 2),
                                collate_fn=SeedDataset.collate_fn
                            )
                            seed_iterator = SeeddirectionalOneShotIterator(seed_dataloader)
                            '''

                            for i in range(0, self.p.seed_steps):
                                #log = kge_model.cross_train_step(self.p, kge_model, optimizer, seed_iterator)
                                log = kge_model.train_step(self.p, kge_model, optimizer, seed_iterator)
                                training_logs.append(log)
                                print('log2:', type(log), log)
                            ##if step % self.p.log_steps == 0:
                            # if i % 1 == 0:

                            #log = kge_model.cross_train_step(self.p, kge_model, optimizer, seed_iterator)

                    if step >= warm_up_steps:
                        current_learning_rate = current_learning_rate / 10
                        logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, kge_model.parameters()),
                            lr=current_learning_rate
                        )
                        warm_up_steps = warm_up_steps * 3

                    if step % self.p.save_checkpoint_steps == 0:
                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps
                        }
                        KGEModel.save_model(self.p, kge_model, optimizer, save_variable_list)

                    if step % self.p.log_steps == 0:
                        metrics = {}
                        for metric in training_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                        KGEModel.log_metrics(self.p, 'Training average', step, metrics)
                        training_logs = []

                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                KGEModel.save_model(self.p, kge_model, optimizer, save_variable_list)

            entity_embedding = kge_model.entity_embedding.detach().cpu().numpy()
            relation_embedding = kge_model.relation_embedding.detach().cpu().numpy()
            for id in self.side_info.id2ent.keys():
                entity_id = self.side_info.ent_old_id2new_id[id]
                entity = self.side_info.id2ent[entity_id]
                new_id = self.side_info.new_ent2id[entity]
                self.ent2embed[id] = entity_embedding[new_id]
            for id in self.side_info.id2rel.keys():
                entity_id = self.side_info.rel_old_id2new_id[id]
                entity = self.side_info.id2rel[entity_id]
                new_id = self.side_info.new_rel2id[entity]
                self.rel2embed[id] = relation_embedding[new_id]
        else:
            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.R_init[id]
        #for id in self.side_info.id2ent.keys(): self.ent2embed[id] = entity_embedding[id]
        #for id in self.side_info.id2rel.keys(): self.rel2embed[id] = relation_embedding[id]

        #for id in self.side_info.id2ent.keys(): self.ent2dict_embed[id] = self.E_init[id]


