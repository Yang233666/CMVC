import pandas as pd
import time
import math
from helper import *
from utils import cos_sim
import numpy as np
from sklearn.cluster import KMeans
from cluster_f1_test import Find_Best_Result, HAC_getClusters, embed2f1, cluster_test, cluster_test_sample, test_sample
# bert fine-tuned
import torch
from torch import nn
from torch import optim
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from find_k_method import Inverse_JumpsMethod

class BertClassificationModel(nn.Module):
    def __init__(self, target_num, max_length):
        super(BertClassificationModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('../data/bert-base-uncased')
        self.bert = BertModel.from_pretrained('../data/bert-base-uncased')
        self.dense = nn.Linear(768, target_num)
        self.max_length = max_length
        print('self.max_length:', self.max_length)

    def __del__(self):
        print("BertClassificationModel del ... ")

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=self.max_length,
                                                           pad_to_max_length=True)  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state)
        return bert_cls_hidden_state, linear_output

class BERT_Model(object):

    def __init__(self, params, side_info, logger, input_list, cluster_predict_list, true_ent2clust, true_clust2ent,
                 model_training_time, BERT_self_training_time, sub_uni2triple_dict=None, rel_id2sentence_list=None, K=0):
        self.p = params
        self.side_info = side_info
        self.logger = logger
        self.input_list = input_list
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.model_training_time = model_training_time
        self.BERT_self_training_time = BERT_self_training_time
        self.sub_uni2triple_dict = sub_uni2triple_dict
        self.rel_id2sentence_list = rel_id2sentence_list
        self.batch_size = 40
        if self.p.dataset == 'reverb45k_change':
            self.epochs = 100
            # self.max_length = 256
        else:
            self.epochs = 120  # 1 2
            # self.epochs = 100
            # self.max_length = 256
            # self.max_length = 64
        # self.epochs = 100  # 1 2
        self.lr = 0.005  # 1 2
        self.K = K
        self.cluster_predict_list =  cluster_predict_list
        # print('self.model_training_time:', self.model_training_time, 'self.BERT_self_training_time:', self.BERT_self_training_time)
        print('self.epochs:', self.epochs)
        # 1 test 125epoch 0.95-0.99
        # 2 test 125epoch 0.94-0.98

        self.coefficient_1, self.coefficient_2 = 0.95, 0.99  # -1  0.95 0.99 is not bad  125-0.005-0.95-0.99-bad  100-0.005-0.95-0.99-3-0.7066  110-0.005-0.95-0.99-3-0.7021-not
        # self.coefficient_1, self.coefficient_2 = 0.96, 0.99  # -2  0.94-0.98-0.703, 125-0.005-0.96-0.99-4-0.705, 100-0.005-0.96-0.99-not  100-0.005-0.95-0.99-3-0.705 120-0.005-0.95-0.99-3-0.7088-ok
        # 0.95 0.98 not, 0.96-0.98 not,

        # all_length = 0
        # for i in range(len(self.input_list)):
        #     if str(self.p.input) == 'entity':
        #         ent_id = self.side_info.ent2id[self.input_list[i]]
        #         if ent_id in self.side_info.isSub:
        #             sentence_id_list = self.side_info.ent_id2sentence_list[ent_id]
        #             longest_index, longest_length = 0, 0
        #             for j in range(len(sentence_id_list)):
        #                 id = sentence_id_list[j]
        #                 sentence = self.side_info.sentence_List[id]
        #                 if len(sentence) > longest_length:
        #                     longest_index, longest_length = j, len(sentence)
        #             all_length += longest_length
        #     else:
        #         rel_id = self.side_info.rel2id[self.input_list[i]]
        #         sentence_id_list = self.rel_id2sentence_list[rel_id]
        #         longest_index, longest_length = 0, 0
        #         for j in range(len(sentence_id_list)):
        #             id = sentence_id_list[j]
        #             sentence = self.side_info.sentence_List[id]
        #             if len(sentence) > longest_length:
        #                 longest_index, longest_length = j, len(sentence)
        #         all_length += longest_length
        # ave = all_length / len(self.input_list)
        # print('all_length:', all_length, 'ave:', ave)
        # self.max_length = int(ave) + 50
        # self.max_length = 128
        self.max_length = 256


    def fine_tune(self):
        folder = 'multi_view/context_view_' + str(self.p.input)
        fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_cls_el_' + str(self.model_training_time) + '_' + str(self.BERT_self_training_time)  # for 1
        # fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_cls_el_' + str(self.model_training_time) + '_' + str(self.BERT_self_training_time) + '_test' # just to test k and epoch for cesi_main_opiec_2

        folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/'
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)

        if not checkFile(fname1):
            print('Fine-tune BERT ', 'self.model_training_time:', self.model_training_time,
                  'self.BERT_self_training_time:', self.BERT_self_training_time, fname1)

            target_list = []
            cluster2target_dict = dict()
            num = 0
            for i in range(len(self.cluster_predict_list)):
                label = self.cluster_predict_list[i]
                if label not in cluster2target_dict:
                    cluster2target_dict.update({label: num})
                    num += 1
                target_list.append(cluster2target_dict[label])
            self.target_num = max(target_list) + 1
            self.sentences_list, self.targets_list = [], []
            self.sub2sentence_id_dict = dict()

            print('self.p.input:', self.p.input)
            print('self.max_length:', self.max_length)
            all_length = 0
            num = 0

            for i in range(len(self.input_list)):
                if str(self.p.input) == 'entity':
                    ent_id = self.side_info.ent2id[self.input_list[i]]
                    if ent_id in self.side_info.isSub:
                        sentence_id_list = self.side_info.ent_id2sentence_list[ent_id]

                        longest_index, longest_length = 0, 0
                        for j in range(len(sentence_id_list)):
                            id = sentence_id_list[j]
                            sentence = self.side_info.sentence_List[id]
                            # if len(sentence) > longest_length and len(sentence) < self.max_length:
                            #     longest_index, longest_length = j, len(sentence)
                            if self.p.input == 1.3:
                                if len(sentence) > longest_length:
                                    longest_index, longest_length = j, len(sentence)
                            else:
                                if len(sentence) > longest_length and len(sentence) < self.max_length + 50:
                                    longest_index, longest_length = j, len(sentence)
                        sentence_id_list = [sentence_id_list[longest_index]]
                        all_length += longest_length
                        sentences_num_list = []
                        for sentence_id in sentence_id_list:
                            sentence = self.side_info.sentence_List[sentence_id]
                            self.sentences_list.append(sentence)
                            target = target_list[i]
                            self.targets_list.append(target)
                            sentences_num_list.append(num)
                            num += 1
                        self.sub2sentence_id_dict.update({i: sentences_num_list})
                else:
                    rel_id = self.side_info.rel2id[self.input_list[i]]
                    sentence_id_list = self.rel_id2sentence_list[rel_id]
                    longest_index, longest_length = 0, 0
                    for j in range(len(sentence_id_list)):
                        id = sentence_id_list[j]
                        sentence = self.side_info.sentence_List[id]
                        # if len(sentence) > longest_length and len(sentence) < self.max_length:
                        #     longest_index, longest_length = j, len(sentence)
                        if self.p.input == 1.3:
                            if len(sentence) > longest_length:
                                longest_index, longest_length = j, len(sentence)
                        else:
                            if len(sentence) > longest_length and len(sentence) < self.max_length + 50:
                                longest_index, longest_length = j, len(sentence)
                    sentence_id_list = [sentence_id_list[longest_index]]
                    all_length += longest_length
                    sentences_num_list = []
                    for sentence_id in sentence_id_list:
                        sentence = self.side_info.sentence_List[sentence_id]
                        self.sentences_list.append(sentence)
                        target = target_list[i]
                        self.targets_list.append(target)
                        sentences_num_list.append(num)
                        num += 1
                    self.sub2sentence_id_dict.update({i: sentences_num_list})
            ave = all_length / len(self.input_list)
            print('all_length:', all_length, 'ave:', ave)
            print()
            print('self.sentences_list:', type(self.sentences_list), len(self.sentences_list))
            print('self.targets_list:', type(self.targets_list), len(self.targets_list), self.targets_list)
            different_labels = list(set(self.targets_list))
            print('different_labels:', type(different_labels), len(different_labels), different_labels)

            sentence_data = {'sentences': self.sentences_list, 'targets': self.targets_list}
            frame = pd.DataFrame(sentence_data)
            self.sentences = frame['sentences'].values
            self.targets = frame['targets'].values

            self.train_inputs, self.train_targets = self.sentences, self.targets
            batch_count = math.ceil(len(self.train_inputs) / self.batch_size)
            print('batch_count:', batch_count)

            batch_train_inputs, batch_train_targets = [], []
            for i in range(batch_count):
                batch_train_inputs.append(self.train_inputs[i * self.batch_size: (i + 1) * self.batch_size])
                batch_train_targets.append(self.train_targets[i * self.batch_size: (i + 1) * self.batch_size])

            # train the model
            bert_classifier_model = BertClassificationModel(self.target_num, self.max_length).cuda()
            optimizer = optim.SGD(bert_classifier_model.parameters(), lr=self.lr)  # real
            # optimizer = optim.Adam(bert_classifier_model.parameters(), lr=self.lr)  # 1_MVC_reverb45k_2
            criterion = nn.CrossEntropyLoss()
            for epoch in range(self.epochs):
                avg_epoch_loss = 0
                for i in range(batch_count):
                    inputs = batch_train_inputs[i]
                    labels = torch.tensor(batch_train_targets[i]).cuda()
                    optimizer.zero_grad()
                    self.bert_cls_hidden_state, outputs = bert_classifier_model(inputs)
                    if epoch == self.epochs - 1:
                    # if epoch == 0:  # test only use bert without fine-tune
                        if i == 0:
                            cls_output = self.bert_cls_hidden_state
                            output_label = outputs.argmax(1)
                        else:
                            cls_output = torch.cat((cls_output, self.bert_cls_hidden_state), 0)
                            output_label = torch.cat((output_label, outputs.argmax(1)), 0)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    avg_epoch_loss += loss.item()
                    if i == (batch_count - 1):
                        real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                        print(real_time, "Epoch: %d, Loss: %.4f" % (epoch, avg_epoch_loss))
                # print('cls_output:', type(cls_output), cls_output.shape, cls_output)
                # print('output_label:', type(output_label), output_label.shape, output_label)
                # print('labels:', type(labels), labels.shape, labels)

                '''测试只用BERT的效果'''
                # cls = cls_output.detach().cpu().numpy()
                # real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                # print(real_time)
                # if self.p.use_first_sentence:
                #     CLS = cls
                # else:
                #     CLS = np.zeros(shape=(len(self.sub2sentence_id_dict), 768))
                #     for i in range(len(self.sub2sentence_id_dict)):
                #         sentences_id_list = self.sub2sentence_id_dict[i]
                #         for id in sentences_id_list:
                #             CLS[i] += cls[id]
                #         CLS[i] /= len(sentences_id_list)
                #
                # best_threshold = dict()
                # best_cluster_threshold, best_ave_f1 = 0, 0
                # cluster_threshold_max, cluster_threshold_min = 60, 0
                # # cluster_threshold_max, cluster_threshold_min = 390, 370
                # for i in tqdm(range(cluster_threshold_max, cluster_threshold_min, -1)):
                #     threshold = i / 100
                #     # threshold = i / 1000
                #     clusters, clusters_center = HAC_getClusters(self.p, CLS, threshold, True)
                #     cluster_predict_list = list(clusters)
                #     if self.p.dataset == 'NYTimes2018':
                #         # 获得NYT数据集的sample 100 clusters的结果
                #         folder_sample = 'multi_view/sample_result/BERT'
                #         print('folder_sample:', folder_sample)
                #         folder_to_make_sample = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/'
                #         if not os.path.exists(folder_to_make_sample):
                #             os.makedirs(folder_to_make_sample)
                #         fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/cesi_clust2ent_u.json'
                #         fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_clust2ent.json'
                #         fname3 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_ent2sentences.json'
                #         if not checkFile(fname1) or not checkFile(fname2):
                #             for i in range(100):
                #                 print('threshold:', threshold)
                #                 model_clust2ent_u, true_clust2ent, ave_f1 = cluster_test_sample(self.p, self.side_info,
                #                                                                                 cluster_predict_list,
                #                                                                                 self.true_ent2clust,
                #                                                                                 self.true_clust2ent,
                #                                                                                 print_or_not=True)
                #                 if float(ave_f1) > 0.55 and float(ave_f1) < 0.59:
                #                     for k in model_clust2ent_u:
                #                         v = list(model_clust2ent_u[k])
                #                         model_clust2ent_u[k] = v
                #                     for k in true_clust2ent:
                #                         v = list(true_clust2ent[k])
                #                         true_clust2ent[k] = v
                #                     model_clust2ent_u_str = json.dumps(model_clust2ent_u, indent=4)
                #                     with open(fname1, 'w') as json_file:
                #                         json_file.write(model_clust2ent_u_str)
                #                     print('dump model_clust2ent_u ok')
                #
                #                     true_clust2ent_str = json.dumps(true_clust2ent, indent=4)
                #                     with open(fname2, 'w') as json_file:
                #                         json_file.write(true_clust2ent_str)
                #                     print('dump true_clust2ent ok')
                #                     true_ent2clust = invertDic(true_clust2ent, 'm2os')
                #                     true_ent2sentences = dict()
                #                     for ent_unique in true_ent2clust:
                #                         trp = self.sub_uni2triple_dict[ent_unique]
                #                         trp_new = dict()
                #                         trp_new['triple'] = trp['triple']
                #                         trp_new['src_sentences'] = trp['src_sentences']
                #                         trp_new['triple_unique'] = trp['triple_unique']
                #                         true_ent2sentences[ent_unique] = trp_new
                #                     true_ent2sentences_str = json.dumps(true_ent2sentences, indent=4)
                #                     with open(fname3, 'w') as json_file:
                #                         json_file.write(true_ent2sentences_str)
                #                     print('dump true_ent2sentences ok')
                #                     exit()
                #                     break
                #         else:
                #             print('load')
                #             f = open(fname1, 'r')
                #             content = f.read()
                #             model_clust2ent_u = json.loads(content)
                #             f.close()
                #             f = open(fname2, 'r')
                #             content = f.read()
                #             true_clust2ent = json.loads(content)
                #             f.close()
                #             test_sample(model_clust2ent_u, true_clust2ent, print_or_not=True)
                #     else:
                #         ave_f1 = embed2f1(self.p, CLS, threshold, self.side_info, self.true_ent2clust,
                #                           self.true_clust2ent,
                #                           dim_is_bert=True)
                #         best_threshold.update({threshold: ave_f1})
                #
                # for cluster_threshold in range(cluster_threshold_max, cluster_threshold_min, -1):
                #     cluster_threshold_real = cluster_threshold / 100
                #     value = best_threshold[cluster_threshold_real]
                #     if value > best_ave_f1:
                #         best_cluster_threshold = cluster_threshold_real
                #         best_ave_f1 = value
                #     else:
                #         continue
                # print(best_threshold)
                #
                # # best_cluster_threshold = 0.05  # from opiec valid for opiec test
                # best_cluster_threshold = 0.15  # from reverb45k valid for reverb45k test
                # print('best_cluster_threshold:', best_cluster_threshold, 'best_ave_f1:', best_ave_f1)
                # clusters, clusters_center = HAC_getClusters(self.p, CLS, best_cluster_threshold, True)
                # cluster_predict_list = list(clusters)
                # ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                # pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                #     = cluster_test(self.p, self.side_info, cluster_predict_list, self.true_ent2clust,
                #                    self.true_clust2ent)
                # print('BERT CLS result:')
                # print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                #       'pair_prec=', pair_prec)
                # print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                #       'pair_recall=', pair_recall)
                # print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
                # print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
                # print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
                # print()
                # exit()
                '''测试只用BERT的效果'''

            self.BERT_CLS = cls_output.detach().cpu().numpy()
            pickle.dump(self.BERT_CLS, open(fname1, 'wb'))
        else:
            print('load fine-tune BERT CLS  ', 'self.model_training_time:', self.model_training_time,
                  'self.BERT_self_training_time:', self.BERT_self_training_time)
            print('self.BERT_CLS:', fname1)
            self.BERT_CLS = pickle.load(open(fname1, 'rb'))

        print('self.BERT_CLS:', type(self.BERT_CLS), self.BERT_CLS.shape)

        # best_threshold = dict()
        # best_cluster_threshold, best_ave_f1 = 0, 0
        # # cluster_threshold_max, cluster_threshold_min = 80, 30  # opiec
        # cluster_threshold_max, cluster_threshold_min = 80, 10
        # minus_num, division_num = -1, 100
        # for i in tqdm(range(cluster_threshold_max, cluster_threshold_min, minus_num)):
        #     threshold = i / division_num
        #     ave_f1 = embed2f1(self.p, self.BERT_CLS, threshold, self.side_info, self.true_ent2clust, self.true_clust2ent,
        #                       dim_is_bert=True, print_or_not=True)
        #     best_threshold.update({threshold: ave_f1})
        # for cluster_threshold in range(cluster_threshold_max, cluster_threshold_min, minus_num):
        #     cluster_threshold_real = cluster_threshold / division_num
        #     value = best_threshold[cluster_threshold_real]
        #     if value > best_ave_f1:
        #         best_cluster_threshold = cluster_threshold_real
        #         best_ave_f1 = value
        #     else:
        #         continue
        # # print(best_threshold)
        # print('best_cluster_threshold:', best_cluster_threshold, 'best_ave_f1:', best_ave_f1)
        # clusters, clusters_center = HAC_getClusters(self.p, self.BERT_CLS, best_cluster_threshold, True)

        # kmeans = KMeans(n_clusters=self.K, n_init=30, max_iter=300, tol=1e-9, n_jobs=10)
        # clusters = kmeans.fit_predict(self.BERT_CLS)


        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # dist = pdist(self.BERT_CLS, metric=self.p.metric)
        # clust_res = linkage(dist, method=self.p.linkage)
        # # for i in range(6000, 8000, 100):
        # for i in range(20, 90, 1):
        #     i = float(i / 100)
        #     print('i:', i)
        #     # clusters = fcluster(clust_res, t=i, criterion='maxclust') - 1
        #     clusters = fcluster(clust_res, t=i, criterion='distance') - 1
        #     cluster_predict_list = list(clusters)
        #     ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        #     pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
        #         = cluster_test(self.p, self.side_info, cluster_predict_list, self.true_ent2clust,
        #                        self.true_clust2ent)
        #     print('self.model_training_time:', self.model_training_time,
        #           'self.BERT_self_training_time:', self.BERT_self_training_time, 'Best BERT CLS result:')
        #     print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
        #           'pair_prec=', pair_prec)
        #     print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
        #           'pair_recall=', pair_recall)
        #     print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        #     print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        #     print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        #     print()

        fname3 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_cls_K_' + str(self.BERT_self_training_time)
        if not checkFile(fname3):
            print('Inverse Jump:')
            K_min, K_max = int(self.K * self.coefficient_1), int(self.K * self.coefficient_2)
            # gap = int((K_max - K_min) / 100) + 1
            gap = int((K_max - K_min) / 20) + 1
            print('K_min:', K_min, 'K_max:', K_max, 'gap:', gap)
            cluster_list = range(K_min, K_max, gap)
            jm = Inverse_JumpsMethod(data=self.BERT_CLS, k_list=cluster_list, dim_is_bert=True)
            jm.Distortions(random_state=0)
            distortions = jm.distortions
            jm.Jumps(distortions=distortions)
            level_one_Inverse_JumpsMethod = jm.recommended_cluster_number
            pickle.dump(level_one_Inverse_JumpsMethod, open(fname3, 'wb'))
        else:
            print('load level_one_Inverse_JumpsMethod:', fname3)
            level_one_Inverse_JumpsMethod = pickle.load(open(fname3, 'rb'))

        # print('new jump level_one_k:', level_one_k)
        print('Inverse_JumpsMethod k:', level_one_Inverse_JumpsMethod)

        dist = pdist(self.BERT_CLS, metric=self.p.metric)
        clust_res = linkage(dist, method=self.p.linkage)
        clusters = fcluster(clust_res, t=level_one_Inverse_JumpsMethod, criterion='maxclust') - 1
        cluster_predict_list = list(clusters)
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            = cluster_test(self.p, self.side_info, cluster_predict_list, self.true_ent2clust,
                           self.true_clust2ent)
        print('self.model_training_time:', self.model_training_time,
              'self.BERT_self_training_time:', self.BERT_self_training_time, 'Best BERT CLS result:')
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()
        return cluster_predict_list, level_one_Inverse_JumpsMethod