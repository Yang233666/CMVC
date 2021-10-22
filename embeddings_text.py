import gensim
from helper import *
#from skge.transe_soft import TransE_soft
from utils import *
import logging
import os

from torch.utils.data import DataLoader

#from dataloader import TrainDataset
#from dataloader import BidirectionalOneShotIterator
#from RotatE_dataloader import *
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
        clean_ent_list, clean_rel_list, clean_sub_list, clean_obj_list = [], [], [], []
        if self.p.use_Entity_linking_dict:
            for ent in self.side_info.new_ent_list: clean_ent_list.append(ent.split('|')[0])
            for rel in self.side_info.new_rel_list: clean_rel_list.append(rel.split('|')[0])
        else:
            for ent in self.side_info.ent_list: clean_ent_list.append(ent.split('|')[0])
            for rel in self.side_info.rel_list: clean_rel_list.append(rel.split('|')[0])
            for sub in self.side_info.sub_list: clean_sub_list.append(sub.split('|')[0])
            for obj in self.side_info.obj_list: clean_obj_list.append(obj.split('|')[0])

        ''' Intialize embeddings '''
        if self.p.embed_init == 'crawl':
            fname1, fname2 = '1E_init', '1R_init'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate init word2vec dict embeddings')
                print('use pre-trained vectors:', self.p.embed_loc)
                model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
                self.E_init = getEmbeddings(model, clean_ent_list, self.p.embed_dims)
                self.R_init = getEmbeddings(model, clean_rel_list, self.p.embed_dims)

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

        if self.p.use_context:
            self.ent2embed = []
            sentence_num, all_num, oov_num, oov_rate = 0, 0, 0, 0
            entity_oov_num, entity_oov_rate = 0, 0
            print('generate text embeddings')
            print('use pre-trained vectors:', self.p.embed_loc)
            model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
            all_sentence_num = len(self.side_info.text_list)
            print('all sentence num:', all_sentence_num)
            start_time = time.time()
            unk = str('UNK')
            if self.p.use_attention:
                print('use attention')
                for i in range(len(self.side_info.text_list)):
                    word_list = self.side_info.text_list[i]
                    ent_embed = self.E_init[i]
                    vec_text = np.zeros(self.p.embed_dims, np.float32)
                    sim_sum = 0
                    #print('i:', i, 'word_list:', len(word_list))
                    for word in word_list:
                        entity_oov_cal_num = 0
                        vec = np.zeros(self.p.embed_dims, np.float32)
                        # print('word:', type(word), word)
                        all_num += 1
                        if word in model.vocab:
                            vec += model.word_vec(word)
                        else:
                            vec += np.random.randn(self.p.embed_dims)
                            #vec += model.word_vec(unk)
                            oov_num += 1
                            entity_oov_cal_num += 1
                        #sim = cos_sim(ent_embed, vec)
                        sim = 0.99 - cos_sim(ent_embed, vec)
                        sim_sum += sim
                        vec_text += sim * vec
                        #print('sim:', sim)
                    #self.ent2embed.append(ent_embed + (normalization2(vec_text / sim_sum)) / 2)
                    self.ent2embed.append(ent_embed + (vec_text / sim_sum) / 2)
                    # print('self.ent2embed:', len(self.ent2embed), self.ent2embed)
                    sentence_num += 1
                    if entity_oov_cal_num > 0:
                        entity_oov_num += 1
                    if sentence_num % 1000 == 0 or sentence_num == all_sentence_num:
                        end_time = time.time()
                        cost_time = end_time - start_time
                        print('sentence_num:', sentence_num, 'cost_time:', cost_time)
                        #for k in range(len(self.ent2embed)):
                            #vec1 = self.ent2embed[k]
                            #vec2 = self.E_init[k]
                            #bias = vec1 - vec2
                            #sum = np.sum(np.array(bias))
                            #print('k:', k, 'bias sum:', sum)
                        #breakpoint()
                oov_rate = oov_num / all_num
                entity_oov_rate = entity_oov_num / all_sentence_num
                print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
                print('entity oov rate:', entity_oov_rate, 'oov num:', entity_oov_num, 'all num:', all_sentence_num)
            else:
                print('do not use attention')
                for i in range(len(self.side_info.text_list)):
                    word_list = self.side_info.text_list[i]
                    ent_embed = self.E_init[i]
                    vec_text = np.zeros(self.p.embed_dims, np.float32)
                    # print('word_list:', len(word_list))
                    for word in word_list:
                        entity_oov_cal_num = 0
                        vec = np.zeros(self.p.embed_dims, np.float32)
                        # print('word:', type(word), word)
                        all_num += 1
                        if word in model.vocab:
                            vec += model.word_vec(word)
                        else:
                            #vec += np.random.randn(self.p.embed_dims)
                            vec += model.word_vec(unk)
                            oov_num += 1
                            entity_oov_cal_num += 1
                        vec_text += vec
                    # print('ent_text_embed_list:', len(ent_text_embed_list))
                    #self.ent2embed.append(normalization((ent_embed) + (normalization(vec_text / len(word_list))) / 2))
                    self.ent2embed.append(ent_embed + (vec_text / len(word_list)) / 2)
                    # print('self.ent2embed:', len(self.ent2embed), self.ent2embed)
                    sentence_num += 1
                    # breakpoint()
                    if entity_oov_cal_num > 0:
                        entity_oov_num += 1
                    if sentence_num % 1000 == 0 or sentence_num == all_sentence_num:
                        end_time = time.time()
                        cost_time = end_time - start_time
                        print('sentence_num:', sentence_num, 'cost_time:', cost_time)
                        # breakpoint()
                oov_rate = oov_num / all_num
                entity_oov_rate = entity_oov_num / all_sentence_num
                print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
                print('entity oov rate:', entity_oov_rate, 'oov num:', entity_oov_num, 'all num:', all_sentence_num)
        else:
            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]


        if self.p.use_Embedding_model:
            if self.p.init_checkpoint:
                KGEModel.override_config(self)

            # Write logs to checkpoint and console
            KGEModel.set_logger(self)

            if self.p.use_Entity_linking_dict:
                nentity, nrelation = len(self.side_info.new_ent_list), len(self.side_info.new_rel_list)
                train_triples = self.side_info.new_trpIds
                # train_sim = [1] * len(train_triples)
            else:
                nentity, nrelation = len(self.side_info.ent_list), len(self.side_info.rel_list)
                train_triples = self.side_info.trpIds
                # train_sim = [1] * len(train_triples)

            self.nentity = nentity
            self.nrelation = nrelation

            logging.info('Model: %s' % self.p.model)
            logging.info('#entity: %d' % nentity)
            logging.info('#relation: %d' % nrelation)
            logging.info('#train: %d' % len(train_triples))

            kge_model = KGEModel(
                model_name=self.p.model,
                dict_local=self.p.embed_loc,
                init=self.p.embed_init,
                E_init=self.E_init,
                R_init=self.R_init,
                nentity=nentity,
                nrelation=nrelation,
                hidden_dim=self.p.hidden_dim,
                gamma=self.p.single_gamma,
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
                    TrainDataset(train_triples, nentity, nrelation, self.p.single_negative_sample_size, 'head-batch'),
                    batch_size=self.p.single_batch_size,
                    shuffle=True,
                    num_workers=max(1, self.p.cpu_num // 2),
                    collate_fn=TrainDataset.collate_fn
                )

                train_dataloader_tail = DataLoader(
                    TrainDataset(train_triples, nentity, nrelation, self.p.single_negative_sample_size, 'tail-batch'),
                    batch_size=self.p.single_batch_size,
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
            logging.info('single_batch_size = %d' % self.p.single_batch_size)
            logging.info('single_negative_adversarial_sampling = %d' % self.p.single_negative_sample_size)
            logging.info('hidden_dim = %d' % self.p.hidden_dim)
            logging.info('single_gamma = %f' % self.p.single_gamma)
            logging.info('negative_adversarial_sampling = %s' % str(self.p.negative_adversarial_sampling))
            if self.p.negative_adversarial_sampling:
                logging.info('adversarial_temperature = %f' % self.p.adversarial_temperature)

            # Set valid dataloader as it would be evaluated during training

            if self.p.do_train:
                logging.info('learning_rate = %d' % current_learning_rate)

                training_logs = []
                if self.p.use_cross_seed:
                    seed_triples = self.side_info.seed_trpIds
                    print('seed_triples:', len(seed_triples))
                    seed_sim = self.side_info.seed_sim
                    logging.info('#seed: %d' % len(seed_triples))

                    # use this when do not update seeds
                    '''
                    seed_dataloader = DataLoader(
                        SeedDataset(seed_triples, nentity, nrelation, seed_sim),
                        batch_size=self.p.batch_size,
                        shuffle=True,
                        num_workers=max(1, self.p.cpu_num // 2),
                        collate_fn=SeedDataset.collate_fn
                    )
                    seed_iterator = SeeddirectionalOneShotIterator(seed_dataloader)
                    '''
                    seed_dataloader_head = DataLoader(
                        SeedDataset(seed_triples, nentity, nrelation, self.p.cross_negative_sample_size, 'head-batch', seed_sim),
                        batch_size=self.p.cross_batch_size,
                        shuffle=True,
                        num_workers=max(1, self.p.cpu_num // 2),
                        collate_fn=SeedDataset.collate_fn
                    )

                    seed_dataloader_tail = DataLoader(
                        SeedDataset(seed_triples, nentity, nrelation, self.p.cross_negative_sample_size, 'tail-batch', seed_sim),
                        batch_size=self.p.cross_batch_size,
                        shuffle=True,
                        num_workers=max(1, self.p.cpu_num // 2),
                        collate_fn=SeedDataset.collate_fn
                    )

                    seed_iterator = BidirectionalOneShotIterator(seed_dataloader_head, seed_dataloader_tail)


                # Training Loop
                for step in range(init_step, self.p.max_steps):

                    log = kge_model.train_step(self.p, kge_model, optimizer, train_iterator)
                    training_logs.append(log)

                    if self.p.use_cross_seed:
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
                                log = kge_model.cross_train_step(self.p, kge_model, optimizer, seed_iterator)
                                #log = kge_model.train_step(self.p, kge_model, optimizer, seed_iterator)
                                training_logs.append(log)
                                #print('log2:', type(log), log)

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

            '''
            fname1, fname2 = '1_train_entity_embedding', '1_train_relation_embedding'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate train embeddings')
                pickle.dump(entity_embedding, open(fname1, 'wb'))
                pickle.dump(relation_embedding, open(fname2, 'wb'))
            else:
                print('load train embeddings')
                entity_embedding = pickle.load(open(fname1, 'rb'))
                relation_embedding = pickle.load(open(fname2, 'rb'))
            '''

            if self.p.use_Entity_linking_dict:
                for id in self.side_info.id2ent.keys():
                    entity_id = self.side_info.ent_old_id2new_id[id]
                    entity = self.side_info.id2ent[entity_id]
                    new_id = self.side_info.new_ent2id[entity]
                    self.ent2embed[id] = entity_embedding[new_id]
                    '''
                    true_id = self.side_info.true_seed_new_id2new_id[new_id]
                    self.ent2embed[id] = entity_embedding[true_id]
                    '''
                for id in self.side_info.id2rel.keys():
                    relation_id = self.side_info.rel_old_id2new_id[id]
                    relation = self.side_info.id2rel[relation_id]
                    new_id = self.side_info.new_rel2id[relation]
                    self.rel2embed[id] = relation_embedding[new_id]
            else:
                for id in self.side_info.id2ent.keys(): self.ent2embed[id] = entity_embedding[id]
                for id in self.side_info.id2rel.keys(): self.rel2embed[id] = relation_embedding[id]


        else:  # do not use embedding model
            if self.p.use_Entity_linking_dict:
                for id in self.side_info.id2ent.keys():
                    entity_id = self.side_info.ent_old_id2new_id[id]
                    entity = self.side_info.id2ent[entity_id]
                    new_id = self.side_info.new_ent2id[entity]
                    self.ent2embed[id] = self.E_init[new_id]
                for id in self.side_info.id2rel.keys():
                    relation_id = self.side_info.rel_old_id2new_id[id]
                    relation = self.side_info.id2rel[relation_id]
                    new_id = self.side_info.new_rel2id[relation]
                    self.rel2embed[id] = self.R_init[new_id]
            else:
                #for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]
                for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.R_init[id]

                #print('self.T_sub_init:', len(self.T_sub_init), 'self.T_obj_init:', len(self.T_obj_init))
                #print('self.E_sub_init:', len(self.E_sub_init), 'self.E_obj_init:', len(self.E_obj_init))
                '''
                self.ent2embed = []

                for id in range(len(self.side_info.text_list)):
                    print('all id:', len(self.side_info.id2ent), len(self.side_info.text_list))
                    if id < 10000:
                        fname1 = '1sentence_embed_array' + str(10000)
                        self.T_init = pickle.load(open(fname1, 'rb'))
                    elif 10000 <= id < 20000:
                        fname1 = '1sentence_embed_array' + str(20000)
                        self.T_init = pickle.load(open(fname1, 'rb'))
                    elif 20000 <= id < 30000:
                        fname1 = '1sentence_embed_array' + str(30000)
                        self.T_init = pickle.load(open(fname1, 'rb'))
                    elif 30000 <= id < 40000:
                        fname1 = '1sentence_embed_array' + str(40000)
                        self.T_init = pickle.load(open(fname1, 'rb'))
                    else:
                        fname1 = '1sentence_embed_array' + str(50724)
                        self.T_init = pickle.load(open(fname1, 'rb'))
                    ent_embed = self.E_init[id]
                    ent_text_embed_list = self.T_init[id]
                    vec = np.zeros(self.p.embed_dims, np.float32)
                    for j in range(len(ent_text_embed_list)):
                        ent_text_embed = ent_text_embed_list[j]
                        sim = cos_sim(ent_embed, ent_text_embed)
                        vec += sim * ent_text_embed
                    self.ent2embed.append((ent_embed) + (vec / len(ent_text_embed_list)) / 2)
                '''
                '''
                for id in self.side_info.id2ent.keys():
                    if id in self.side_info.isSub.keys():
                        ent_embed = self.E_init[id]
                        sub_id = self.side_info.id2sub(self.side_info.id2ent[id])
                        ent_text_embed_list = self.T_sub_init[sub_id]
                        vec = np.zeros(self.p.embed_dims, np.float32)
                        for j in range(len(ent_text_embed_list)):
                            ent_text_embed = ent_text_embed_list[j]
                            sim = cos_sim(ent_embed, ent_text_embed)
                            vec += sim * ent_text_embed
                        self.ent2embed.append((ent_embed) + (vec / len(ent_text_embed_list)) / 2)
                    else:
                        ent_embed = self.E_init[id]
                        obj_id = self.side_info.id2obj(self.side_info.id2ent[id])
                        ent_text_embed_list = self.T_obj_init[obj_id]
                        vec = np.zeros(self.p.embed_dims, np.float32)
                        for j in range(len(ent_text_embed_list)):
                            ent_text_embed = ent_text_embed_list[j]
                            sim = cos_sim(ent_embed, ent_text_embed)
                            vec += sim * ent_text_embed
                        self.ent2embed.append((ent_embed) + (vec / len(ent_text_embed_list)) / 2)
                    '''
        #for id in self.side_info.id2ent.keys(): self.ent2embed[id] = entity_embedding[id]
        #for id in self.side_info.id2rel.keys(): self.rel2embed[id] = relation_embedding[id]

        #for id in self.side_info.id2ent.keys(): self.ent2dict_embed[id] = self.E_init[id]


