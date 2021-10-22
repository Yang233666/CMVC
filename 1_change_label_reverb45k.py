from helper import *
from utils import *
# reload(sys);
# sys.setdefaultencoding('utf-8')			# Swtching from ASCII to UTF-8 encoding

''' *************************************** DATASET PREPROCESSING **************************************** '''
# Obama, Apple, Fox, Joe Biden, Windows, IBM, Sony, Bush, HTML, Microsoft, AMD, Internet Explorer
# Pakistan, Wikipedia->Wiki, Google,
# Interstate 10, Interstate 20, Interstate 80, Interstate 90

class CESI_Main(object):

    def __init__(self, args):
        self.p = args
        self.logger = getLogger(args.name, args.log_dir, args.config_dir)
        self.logger.info('Running {}'.format(args.name))
        self.read_triples()


    def read_triples(self):
        self.logger.info('Reading Triples')

        fname = self.p.out_path + self.p.file_triples  # File for storing processed triples
        self.triples_list = []  # List of all triples in the dataset
        self.amb_ent = ddict(int)  # Contains ambiguous entities in the dataset
        self.amb_mentions = {}  # Contains all ambiguous mentions
        self.isAcronym = {}  # Contains all mentions which can be acronyms

        self.origin_triples_list = []  # 数据集原版的triple，直接在这个上面修改
        sub2true_link = dict()
        true_link_and_ent2triple_id = dict()
        # triple_id_num = 0
        print('dataset:', args.dataset)
        if args.dataset == 'OPIEC':
            print('load OPIEC_dataset ... ')
            self.triples_list = pickle.load(open(args.data_path, 'rb'))

            fname1 = args.data_dir + '/' + 'OPIEC/' + 'OPIEC_53k_final'
            print('fname1:', fname1)
            with open(fname1, 'w') as f:
                f.write('\n'.join([json.dumps(triple) for triple in self.triples_list]))
            f.close()
            exit()

            ''' Ground truth clustering '''
            self.true_ent2clust = ddict(set)
            for trp in self.triples_list:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust[sub_u].add(trp['subject_wiki_link'])
            self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')

        else:
            if not checkFile(fname):
                with codecs.open(args.data_path, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        trp = json.loads(line.strip())
                        origin_trp = trp.copy()
                        self.origin_triples_list.append(origin_trp)

                        trp['raw_triple'] = trp['triple']
                        sub, rel, obj = map(str, trp['triple'])

                        '''
                        if sub.isalpha() and sub.isupper(): self.isAcronym[
                            proc_ent(sub)] = 1  # Check if the subject is an acronym
                        if obj.isalpha() and obj.isupper(): self.isAcronym[
                            proc_ent(obj)] = 1  # Check if the object  is an acronym

                        sub, rel, obj = proc_ent(sub), trp['triple_norm'][1], proc_ent(
                            obj)  # Get Morphologically normalized subject, relation, object

                        # for reverb45k_test_new dataset
                        sub, rel, obj = str(sub).lower(), str(rel).lower(), str(obj).lower()
                        '''

                        if len(sub) == 0 or len(rel) == 0 or len(obj) == 0: continue  # Ignore incomplete triples

                        trp['triple'] = [sub, rel, obj]
                        sub_unique, obj_unique = sub + '|' + str(trp['_id']), obj + '|' + str(trp['_id'])
                        trp['triple_unique'] = [sub + '|' + str(trp['_id']), rel + '|' + str(trp['_id']),
                                                obj + '|' + str(trp['_id'])]
                        trp['ent_lnk_sub'] = trp['entity_linking']['subject']
                        trp['ent_lnk_obj'] = trp['entity_linking']['object']
                        trp['true_sub_link'] = trp['true_link']['subject']
                        trp['true_obj_link'] = trp['true_link']['object']
                        trp['rel_info'] = trp['kbp_info']  # KBP side info for relation

                        self.triples_list.append(trp)

                        # if sub not in sub2true_link:
                        #     sub2true_link.update({sub: [trp['true_sub_link']]})
                        # else:
                        #     if trp['true_sub_link'] not in sub2true_link[sub]:
                        #         sub2true_link[sub].append(trp['true_sub_link'])
                        #
                        # if trp['true_sub_link'] not in true_link_and_ent2triple_id:
                        #     true_link_and_ent2triple_id.update({trp['true_sub_link']: {sub: [triple_id_num]}})
                        # else:
                        #     if sub not in true_link_and_ent2triple_id[trp['true_sub_link']]:
                        #         true_link_and_ent2triple_id[trp['true_sub_link']].update({sub: [triple_id_num]})
                        #     else:
                        #         if triple_id_num not in true_link_and_ent2triple_id[trp['true_sub_link']][sub]:
                        #             true_link_and_ent2triple_id[trp['true_sub_link']][sub].append(triple_id_num)
                        #
                        # triple_id_num += 1

                with open(fname, 'w') as f:
                    f.write('\n'.join([json.dumps(triple) for triple in self.triples_list]))
                    self.logger.info('\tCached triples')
            else:
                self.logger.info('\tLoading cached triples')
                with open(fname) as f:
                    self.triples_list = [json.loads(triple) for triple in f.read().split('\n')]

        ''' Identifying ambiguous entities '''
        amb_clust = {}
        for trp in self.triples_list:
            sub = trp['triple'][0]
            for tok in sub.split():
                amb_clust[tok] = amb_clust.get(tok, set())
                amb_clust[tok].add(sub)

        for rep, clust in amb_clust.items():
            if rep in clust and len(clust) >= 3:
                self.amb_ent[rep] = len(clust)
                for ele in clust: self.amb_mentions[ele] = 1

        print('self.triples_list:', type(self.triples_list), len(self.triples_list))

        split = 'change'
        folder = args.data_dir + '/' + 'reverb45k_' + str(split) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        fname = folder + 'reverb45k_'  + str(split) + '_test' + '_read'
        print('folder:', folder)
        print('fname:', fname)
        if checkFile(fname):
            self.origin_triples_list = []
            print('load ', fname)
            with open(fname) as f:
                self.origin_triples_list = [json.loads(triple) for triple in f.read().split('\n')]
        print('self.origin_triples_list:', type(self.origin_triples_list), len(self.origin_triples_list))

        ''' Ground truth clustering '''
        self.true_ent2clust = ddict(set)
        for trp in self.origin_triples_list:
            sub, rel, obj = map(str, trp['triple'])
            sub_u = sub + '|' + str(trp['_id'])
            self.true_ent2clust[sub_u].add(trp['true_link']['subject'])
        self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')
        print('self.true_clust2ent:', type(self.true_clust2ent), len(self.true_clust2ent))
        print('self.true_ent2clust:', type(self.true_ent2clust), len(self.true_ent2clust))

        triple_id_num = 0
        for trp in self.origin_triples_list:

            sub, rel, obj = map(str, trp['triple'])

            if len(sub) == 0 or len(rel) == 0 or len(obj) == 0: continue  # Ignore incomplete triples

            trp['triple_unique'] = [sub + '|' + str(trp['_id']), rel + '|' + str(trp['_id']),
                                    obj + '|' + str(trp['_id'])]

            trp['true_sub_link'] = trp['true_link']['subject']

            if sub not in sub2true_link:
                sub2true_link.update({sub: [trp['true_sub_link']]})
            else:
                if trp['true_sub_link'] not in sub2true_link[sub]:
                    sub2true_link[sub].append(trp['true_sub_link'])

            if trp['true_sub_link'] not in true_link_and_ent2triple_id:
                true_link_and_ent2triple_id.update({trp['true_sub_link']: {sub: [triple_id_num]}})
            else:
                if sub not in true_link_and_ent2triple_id[trp['true_sub_link']]:
                    true_link_and_ent2triple_id[trp['true_sub_link']].update({sub: [triple_id_num]})
                else:
                    if triple_id_num not in true_link_and_ent2triple_id[trp['true_sub_link']][sub]:
                        true_link_and_ent2triple_id[trp['true_sub_link']][sub].append(triple_id_num)

            triple_id_num += 1

        num = 0
        for k in sub2true_link:
            v = sub2true_link[k]
            num += len(v)
        print('sub2true_link:', type(sub2true_link), len(sub2true_link), 'num:', num)

        num = 0
        for k in true_link_and_ent2triple_id:
            for v1 in true_link_and_ent2triple_id[k]:
                v2 = true_link_and_ent2triple_id[k][v1]
                num += len(v2)
        true_link_num = len(true_link_and_ent2triple_id)
        print('true_link_and_ent2triple_id:', type(true_link_and_ent2triple_id), true_link_num, 'num:', num)

        true_link2triples_list_id = dict()
        for i in range(len(self.triples_list)):
            triple = self.triples_list[i]
            sub_true_link = triple['true_link']['subject']
            if sub_true_link not in true_link2triples_list_id:
                true_link2triples_list_id.update({sub_true_link: [i]})
            else:
                true_link2triples_list_id[sub_true_link].append(i)
        print('true_link2triples_list_id:', type(true_link2triples_list_id), len(true_link2triples_list_id))

        folder_to_make = '../file/' + 'get_reverb45k_label'
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)
        fname1 = folder_to_make + '/' + 'wrong_clust'
        fname2 = folder_to_make + '/' + 'wrong_triple'
        fname3 = folder_to_make + '/' + 'clust_order'
        fname4 = folder_to_make + '/' + 'clust_num_and_new_true_link_num_list'
        print('fname1:', fname1)
        print('fname2:', fname2)
        print('fname3:', fname3)
        print('fname4:', fname4)
        if not checkFile(fname1):
            print('generate ', fname1)
            wrong_clust, wrong_triple = [], []
            clust_num_and_new_true_link_num_list = []
            clust_num = 0
            new_true_link_num = 10000
        else:
            print('load ', fname1)
            wrong_clust = pickle.load(open(fname1, 'rb'))
            wrong_triple = pickle.load(open(fname2, 'rb'))

            clust_num_and_new_true_link_num_list = pickle.load(open(fname4, 'rb'))
            clust_num, new_true_link_num = clust_num_and_new_true_link_num_list[0], clust_num_and_new_true_link_num_list[1]
        wrong_clust, wrong_triple = list(set(wrong_clust)), list(set(wrong_triple))
        print('wrong_clust:', type(wrong_clust), len(wrong_clust))
        print('wrong_triple:', type(wrong_triple), len(wrong_triple))
        print('clust_num_and_new_true_link_num_list:', len(clust_num_and_new_true_link_num_list))
        print('clust_num:', clust_num, 'new_true_link_num:', new_true_link_num)
        exit()

        if not checkFile(fname3):
            print('generate ', fname3)
            clust_order = []
            length2clust = dict()
            for clust in self.true_clust2ent:
                length = len(self.true_clust2ent[clust])
                if length not in length2clust.keys():
                    length2clust.update({length: [clust]})
                else:
                    length2clust[length].append(clust)
            print('length2clust:', type(length2clust), len(length2clust), length2clust)

            length_list = list(length2clust.keys())
            length_list.sort(reverse=True)
            print('length_list:', type(length_list), len(length_list), length_list)

            for i in range(len(length_list)):
                length = length_list[i]
                clust_list = length2clust[length]
                for j in range(len(clust_list)):
                    clust = clust_list[j]
                    clust_order.append(clust)
            pickle.dump(clust_order, open(fname3, 'wb'))
        else:
            print('load ', fname3)
            clust_order = pickle.load(open(fname3, 'rb'))
        print('clust_order:', type(clust_order), len(clust_order))

        stop = 0
        search_num = 0
        for i in range(len(clust_order)):
            if i == clust_num or i > clust_num:
                clust = clust_order[i]
            # elif search_num < 20:
            #     print('search_num:', search_num)
            #     print('请输入要查询的clust；若输入0则表示停止:')
            #     clust = input()
            #     search_num += 1
            else:
                clust = '/m/055yr'
                stop = 1
                print('停止')
                break

            if stop == 1:
                print('停止')
                break
            ent_id_set = self.true_clust2ent[clust]
            ent_list = list()
            for ent_id in list(ent_id_set):
                ent = ent_id.split('|')[0]
                # ent = ent_id
                if ent not in ent_list:
                    ent_list.append(ent)
            print('===============================================================================================')
            print('进度：', clust_num + 1, '/', true_link_num, '簇', 'clust:', type(clust), clust)
            print('ent_list:', len(ent_list), ent_list)
            print()
            print('请输入错误的ent，输入1则表示无错误，继续执行下一个；若输入0则表示停止。')
            wrong_ent = input()
            if wrong_ent == str(0):
                stop = 1
                break
            elif wrong_ent == str(1):
                clust_num += 1
                continue
            else:
                print('wrong_ent:', wrong_ent)
                if wrong_ent == str(1):
                    continue
                while wrong_ent != str(0) and wrong_ent != str(1):
                    while wrong_ent not in sub2true_link:
                        print('请重新输入错误的ent；若输入0则表示停止：')
                        wrong_ent = input()
                        if wrong_ent == str(0):
                            stop = 1
                            break
                    triple_id_list = true_link_and_ent2triple_id[clust][wrong_ent]
                    for triple_id in triple_id_list:
                        triple = self.origin_triples_list[triple_id]
                        src_sentences_list = triple['src_sentences']
                        for src_sentences in src_sentences_list:
                            print('wrong_ent src_sentences:', src_sentences)
                    print()
                    true_link_list = sub2true_link[wrong_ent]
                    print('--------------------------------------------------')
                    for true_link in true_link_list:
                        new_ent_list = list()
                        if true_link in self.true_clust2ent:
                            for ent_id in list(self.true_clust2ent[true_link]):
                                ent = ent_id.split('|')[0]
                                if ent not in new_ent_list:
                                    new_ent_list.append(ent)
                            print('true_link:', true_link)
                            print('ent_list:', len(new_ent_list), new_ent_list)
                            print()
                    print('--------------------------------------------------')
                    print('请输入ent的新true_link，输入new则表示建立新的true_link；若有疑问则输入1，若输入0则取消此ent：')
                    new_true_link = input()
                    while new_true_link == str(1):
                        print('请输入有疑问的true_link，若输入0则无疑问：')
                        question_true_link = input()
                        while question_true_link != str(0):
                            print('请输入有疑问的ent，若输入0则无疑问：')
                            question_ent = input()
                            while question_ent != str(0):
                                question_triple_id_list = true_link_and_ent2triple_id[question_true_link][
                                    question_ent]
                                for question_triple_id in question_triple_id_list:
                                    question_triple = self.origin_triples_list[question_triple_id]
                                    src_sentences_list = question_triple['src_sentences']
                                    for src_sentences in src_sentences_list:
                                        print('src_sentences:', src_sentences)
                                print('请输入有疑问的ent，若输入0则无疑问：')
                                question_ent = input()
                            print('请输入有疑问的true_link，若输入0则无疑问：')
                            question_true_link = input()
                        print('请输入ent的新true_link，若有疑问则输入1：')
                        new_true_link = input()
                    if new_true_link != str(0) and new_true_link != 'new':
                        while new_true_link not in self.true_clust2ent:
                            print('请重新输入新true_link；若输入0则表示停止：')
                            new_true_link = input()
                            if new_true_link == str(0):
                                stop = 1
                                break
                    if new_true_link == 'new':
                        new_true_link = '/m/' + str(new_true_link_num)
                        new_true_link_num += 1
                    if new_true_link != str(0) and new_true_link != 'new':
                        triple_id_list = true_link_and_ent2triple_id[clust][wrong_ent]
                        for triple_id in triple_id_list:
                            triple = self.origin_triples_list[triple_id]
                            # print('triple_id:', triple_id)
                            sub, rel, obj = map(str, triple['triple'])
                            true_sub_link = triple['true_link']['subject']
                            old = self.origin_triples_list[triple_id]['true_link']['subject']
                            # print('sub:', sub)
                            # print('true_sub_link:', true_sub_link)
                            # print('old:', )
                            if sub == wrong_ent and clust == true_sub_link:
                                self.origin_triples_list[triple_id]['true_link']['subject'] = new_true_link
                                new = self.origin_triples_list[triple_id]['true_link']['subject']
                                # print('new:', )
                                if clust not in wrong_clust:
                                    wrong_clust.append(clust)
                                if triple_id not in wrong_triple:
                                    wrong_triple.append(triple_id)
                                if wrong_ent in ent_list:
                                    ent_list.pop(ent_list.index(wrong_ent))
                                ent_id_list = list(self.true_clust2ent[clust]).copy()
                                for ent_id in ent_id_list:
                                    if ent_id.split('|')[0] == wrong_ent:
                                        self.true_clust2ent[clust].remove(ent_id)
                                print('成功！', old, ' -> ', new, 'wrong_clust:', len(wrong_clust), 'wrong_triple:',
                                      len(wrong_triple))
                                # print('self.origin_triples_list[triple_id]:', self.origin_triples_list[triple_id]['true_link']['subject'])
                                print()
                                continue
                            else:
                                stop = 1
                                break
                    ent_id_set = self.true_clust2ent[clust]
                    ent_list = list()
                    for ent_id in list(ent_id_set):
                        ent = ent_id.split('|')[0]
                        if ent not in ent_list:
                            ent_list.append(ent)
                    print('ent_list:', len(ent_list), ent_list)
                    print()
                    print('请继续输入错误的ent；输入1则表示无错误，继续执行下一个；若输入0则表示停止：')
                    wrong_ent = input()
                clust_num += 1
            # 生成新的self.true_clust2ent
            self.true_ent2clust = ddict(set)
            for trp in self.origin_triples_list:
                sub, rel, obj = map(str, trp['triple'])
                sub_u = sub + '|' + str(trp['_id'])
                self.true_ent2clust[sub_u].add(trp['true_link']['subject'])
            self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')
        print('wrong_clust:', len(wrong_clust), 'wrong_triple:', len(wrong_triple))
        print('结束啦')
        print('保存文件！')
        split = 'change'
        folder = args.data_dir + '/' + 'reverb45k_' + str(split) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        fname = folder + 'reverb45k_'  + str(split) + '_test'
        print('folder:', folder)
        print('fname:', fname)
        if not checkFile(fname):
            print('generate:', fname)
            pickle.dump(self.origin_triples_list, open(fname, 'wb'))
        else:
            print('load new_triples_list:', fname)
            self.origin_triples_listt = pickle.load(open(fname, 'rb'))
        print('self.origin_triples_list:', type(self.origin_triples_list), len(self.origin_triples_list))

        clust_num_and_new_true_link_num_list = []
        clust_num_and_new_true_link_num_list.append(clust_num)
        clust_num_and_new_true_link_num_list.append(new_true_link_num)
        pickle.dump(wrong_clust, open(fname1, 'wb'))
        pickle.dump(wrong_triple, open(fname2, 'wb'))
        pickle.dump(clust_num_and_new_true_link_num_list, open(fname4, 'wb'))

        fname = fname + '_read'
        print('fname:', fname)
        with open(fname, 'w') as f:
            f.write('\n'.join([json.dumps(triple) for triple in self.origin_triples_list]))
        f.close()
        print('bye')
        exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information')
    parser.add_argument('-data', dest='dataset', default='reverb45k', help='Dataset to run CESI on:base,ambiguous,reverb45k')
    parser.add_argument('-split', dest='split', default='test_new', help='Dataset split for evaluation')
    # parser.add_argument('-data', dest='dataset', default='OPIEC', help='Dataset to run CESI on')
    # parser.add_argument('-split', dest='split', default='53k', help='Dataset split for evaluation')
    parser.add_argument('-data_dir', dest='data_dir', default='../data', help='Data directory')
    parser.add_argument('-out_dir', dest='out_dir', default='../output', help='Directory to store CESI output')
    parser.add_argument('-config_dir', dest='config_dir', default='../config', help='Config directory')
    parser.add_argument('-log_dir', dest='log_dir', default='../log', help='Directory for dumping log files')
    parser.add_argument('-ppdb_url', dest='ppdb_url', default='http://localhost:9997/',
                        help='Address of PPDB server')
    # parser.add_argument('-reset', dest="reset", action='store_true',
    #                     help='Clear the cached files (Start a fresh run)')
    parser.add_argument('-reset', dest="reset", action='store_true', default=True,
                        help='Clear the cached files (Start a fresh run)')
    parser.add_argument('-name', dest='name', default=None, help='Assign a name to the run')
    parser.add_argument('-word2vec_path', dest='word2vec_path', default='../init_dict/crawl-300d-2M.vec', help='word2vec_path')
    parser.add_argument('-alignment_module', dest='alignment_module', default='swapping', help='alignment_module')
    parser.add_argument('-Entity_linking_dict_loc', dest='Entity_linking_dict_loc',
                        default='../init_dict/Entity_linking_dict/Whole_Ranked_Merged_Current_dictionary_UTF-8.txt',
                        help='Location of Entity_linking_dict to be loaded')
    parser.add_argument('-change_EL_threshold', dest='change_EL_threshold', default=False, help='change_EL_threshold')
    parser.add_argument('-entity_EL_threshold', dest='entity_EL_threshold', default=0, help='entity_EL_threshold')
    parser.add_argument('-relation_EL_threshold', dest='relation_EL_threshold', default=0, help='relation_EL_threshold')

    # system settings
    parser.add_argument('-embed_init', dest='embed_init', default='crawl', choices=['crawl', 'random'],
                        help='Method for Initializing NP and Relation embeddings')
    parser.add_argument('-embed_loc', dest='embed_loc', default='../init_dict/crawl-300d-2M.vec',
                        help='Location of embeddings to be loaded')

    parser.add_argument('--use_assume', default=True)
    parser.add_argument('--use_Entity_linking_dict', default=True)

    parser.add_argument('--use_Embedding_model', default=True)
    parser.add_argument('--use_cluster_learning', default=False)
    parser.add_argument('--use_cross_seed', default=False)
    parser.add_argument('--update_seed', default=False)

    parser.add_argument('--use_bert_update_seeds', default=False)
    parser.add_argument('--use_new_embedding', default=False)
    # crawl + TransE + new seed + update seed

    parser.add_argument('--max_steps', default=50000, type=int)
    parser.add_argument('--turn_to_seed', default=2000, type=int)
    parser.add_argument('--seed_max_steps', default=2000, type=int)
    parser.add_argument('--update_seed_steps', default=50000, type=int)

    parser.add_argument('--get_new_cross_seed', default=False)
    parser.add_argument('--entity_threshold', dest='entity_threshold', default=0.9, help='entity_threshold')
    parser.add_argument('--relation_threshold', dest='relation_threshold', default=0.95, help='relation_threshold')

    parser.add_argument('--use_context', default=True)
    parser.add_argument('--use_attention', default=True)
    parser.add_argument('--replace_h', default=True)
    parser.add_argument('--sentence_delete_stopwords', default=True)
    parser.add_argument('--use_first_sentence', default=True)
    parser.add_argument('--use_BERT', default=True)

    # Multi-view
    parser.add_argument('--step_0_use_hac', default=False)

    # HAN
    #parser.add_argument('--data_path', type=str, default='./sample_text.csv')
    #parser.add_argument('--min_word_count', type=int, default=5)

    # parser.add_argument('--epochs', type=int, default=300)
    # parser.add_argument('--batch_size', type=int, default=50)
    # parser.add_argument("--device", default="/gpu:0")
    # parser.add_argument("--lr", type=float, default=0.001)

    # RotatE
    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data', default=False)
    parser.add_argument('--save_path', default='../output', type=str)

    # parser.add_argument('--model', default='new_rotate', type=str)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true', default=False)
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true', default=False)

    parser.add_argument('-n1', '--single_negative_sample_size', default=32, type=int)
    # parser.add_argument('-n1', '--single_negative_sample_size', default=2, type=int)
    parser.add_argument('-n2', '--cross_negative_sample_size', default=40, type=int)
    parser.add_argument('-d', '--hidden_dim', default=300, type=int)
    parser.add_argument('-g1', '--single_gamma', default=12.0, type=float)
    parser.add_argument('-g2', '--cross_gamma', default=0.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b1', '--single_batch_size', default=2048, type=int)
    # parser.add_argument('-b1', '--single_batch_size', default=48, type=int)
    parser.add_argument('-b2', '--cross_batch_size', default=2048, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec', default=True)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=12, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('-embed_dims', dest='embed_dims', default=300, type=int, help='Embedding dimension')

    # word2vec and iteration hyper-parameters
    parser.add_argument('-retrain_literal_embeds', dest='retrain_literal_embeds', default=True,
                        help='retrain_literal_embeds')

    # Clustering hyper-parameters
    parser.add_argument('-linkage', dest='linkage', default='complete', choices=['complete', 'single', 'average'],
                        help='HAC linkage criterion')
    # parser.add_argument('-thresh_val', dest='thresh_val', default=.4239, type=float, help='Threshold for clustering')
    # parser.add_argument('-thresh_val', dest='thresh_val', default=cluster_threshold_real, type=float,
                        #help='Threshold for clustering')
    parser.add_argument('-metric', dest='metric', default='cosine',
                        help='Metric for calculating distance between embeddings')
    parser.add_argument('-num_canopy', dest='num_canopy', default=1, type=int,
                        help='Number of caponies while clustering')
    parser.add_argument('-true_seed_num', dest='true_seed_num', default=2361, type=int)
    args = parser.parse_args()

    # if args.name == None: args.name = args.dataset + '_' + args.split + '_' + time.strftime(
    #     "%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
    if args.name == None: args.name = args.dataset + '_' + args.split + '_' + '1'

    args.file_triples = '/triples.txt'  # Location for caching triples
    args.file_entEmbed = '/embed_ent.pkl'  # Location for caching learned embeddings for noun phrases
    args.file_relEmbed = '/embed_rel.pkl'  # Location for caching learned embeddings for relation phrases
    args.file_entClust = '/cluster_ent.txt'  # Location for caching Entity clustering results
    args.file_relClust = '/cluster_rel.txt'  # Location for caching Relation clustering results
    args.file_sideinfo = '/side_info.txt'  # Location for caching side information extracted for the KG (for display)
    args.file_sideinfo_pkl = '/side_info.pkl'  # Location for caching side information extracted for the KG (binary)
    args.file_hyperparams = '/hyperparams.json'  # Location for loading hyperparameters
    args.file_results = '/results.json'  # Location for loading hyperparameters

    args.out_path = args.out_dir + '/' + args.name  # Directory for storing output
    print('args.log_dir:', args.log_dir)
    print('args.out_path:', args.out_path)
    print('args.reset:', args.reset)
    args.data_path = args.data_dir + '/' + args.dataset + '/' + args.dataset + '_' + args.split  # Path to the dataset
    if args.reset: os.system('rm -r {}'.format(args.out_path))  # Clear cached files if requeste
    if not os.path.isdir(args.out_path): os.system(
        'mkdir -p ' + args.out_path)  # Create the output directory if doesn't exist

    cesi = CESI_Main(args)  # Loading KG triples