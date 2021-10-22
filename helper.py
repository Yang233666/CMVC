import os, sys, re, pdb, time, argparse, logging, logging.config
# import numpy as np, requests, json, operator, pickle, codecs
import numpy as np, json, operator, pickle, codecs
from numpy.fft import fft, ifft
from nltk.tokenize import sent_tokenize, word_tokenize
import itertools, pathlib
from pprint import pprint

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from gensim.utils import lemmatize
from nltk.wsd import lesk
from collections import defaultdict as ddict
from joblib import Parallel, delayed
import gc


def mergeList(list_of_list):
    return list(itertools.chain.from_iterable(list_of_list))


def unique(l):
    return list(set(l))


def checkFile(filename):
    return pathlib.Path(filename).is_file()


def invertDic(my_map, struct='o2o'):
    inv_map = {}

    if struct == 'o2o':  # Reversing one-to-one dictionary
        for k, v in my_map.items():
            inv_map[v] = k

    elif struct == 'm2o':  # Reversing many-to-one dictionary
        for k, v in my_map.items():
            inv_map[v] = inv_map.get(v, [])
            inv_map[v].append(k)

    elif struct == 'm2ol':  # Reversing many-to-one list dictionary
        for k, v in my_map.items():
            for ele in v:
                inv_map[ele] = inv_map.get(ele, [])
                inv_map[ele].append(k)

    elif struct == 'm2os':
        for k, v in my_map.items():
            for ele in v:
                inv_map[ele] = inv_map.get(ele, set())
                inv_map[ele].add(k)

    elif struct == 'ml2o':  # Reversing many_list-to-one dictionary
        for k, v in my_map.items():
            for ele in v:
                inv_map[ele] = inv_map.get(ele, [])
                inv_map[ele] = k
    return inv_map


def dumpCluster(fname, rep2clust, id2name):
    with open(fname, 'w') as f:
        for rep, clust in rep2clust.items():
            f.write(id2name[rep] + '\n')
            for ele in clust:
                f.write('\t' + id2name[ele] + '\n')


def loadCluster(fname, name2id):
    rep2clust = ddict(list)
    with open(fname) as f:
        for line in f:
            if not line.startswith('\t'):
                rep = name2id[line.strip()]
            else:
                rep2clust[rep].append(name2id[line.strip()])

    return rep2clust


# Get embedding of words from gensim word2vec model
def getEmbeddings(model, phr_list, embed_dims):
    embed_list = []
    all_num, oov_num, oov_rate = 0, 0, 0
    for phr in phr_list:
        if phr in model.vocab:
            embed_list.append(model.word_vec(phr))
            all_num += 1
        else:
            vec = np.zeros(embed_dims, np.float32)
            wrds = word_tokenize(phr)
            for wrd in wrds:
                all_num += 1
                if wrd in model.vocab:
                    vec += model.word_vec(wrd)
                else:
                    vec += np.random.randn(embed_dims)
                    oov_num += 1
            if len(wrds) == 0:
                embed_list.append(vec / 10000)
            else:
                embed_list.append(vec / len(wrds))
    oov_rate = oov_num / all_num
    print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
    return np.array(embed_list)


def look_up_Embeddings(model, vocab_list, embed_dims):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    embed_list = []
    OOV_list = []
    all_num, oov_num, oov_rate = 0, 0, 0
    lower_in_num, stem_in_num = 0, 0
    for vocab in vocab_list:
        all_num += 1
        if vocab in model.vocab:
            embed_list.append(model.word_vec(vocab))
        else:
            vocab_lower = vocab.lower()
            if vocab_lower in model.vocab:
                embed_list.append(model.word_vec(vocab_lower))
                lower_in_num += 1
            else:
                vocab_stem = stemmer.stem(vocab)
                if vocab_stem in model.vocab:
                    embed_list.append(model.word_vec(vocab_stem))
                    stem_in_num += 1
                else:
                    embed_list.append(np.random.randn(embed_dims))
                    oov_num += 1
                    OOV_list.append(vocab)

    oov_rate = oov_num / all_num
    print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
    print('in_num:', all_num - oov_num, 'lower_in_num:', lower_in_num, 'stem_in_num:', stem_in_num)
    f = open('./file/look_up_dict_oov.txt', 'w')
    for i in range(len(OOV_list)):
        f.write(str(OOV_list[i]))
        f.write('\n')
    f.close()
    return np.array(embed_list)


def get_text_Embeddings(model, sentence_list, embed_dims):
    sentence_embed_list, word_embed_list = [], []
    sentence_num, all_num, oov_num, oov_rate = 0, 0, 0, 0
    all_sentence_num = len(sentence_list)
    print('all sentence num:', all_sentence_num)
    for word_list in sentence_list:
        #print('word_list:', type(word_list), word_list)
        for word in word_list:
            vec = np.zeros(embed_dims, np.float32)
            #print('word:', type(word), word)
            all_num += 1
            if word in model.vocab:
                vec += model.word_vec(word)
            else:
                vec += np.random.randn(embed_dims)
                oov_num += 1
            word_embed_list.append(vec)
            #print('word_embed_list:', word_embed_list)
        sentence_embed_list.append(np.array(word_embed_list))
        #print('sentence_embed_list:', sentence_embed_list)
        sentence_num += 1
        if sentence_num % 500 == 0 or sentence_num == all_sentence_num:
            print('sentence_num:', sentence_num, 'all sentence num:', all_sentence_num)
            sentence_embed_array = np.array(sentence_embed_list)
            fname1 = '1sentence_embed_array' + str(sentence_num)
            pickle.dump(sentence_embed_array, open(fname1, 'wb'))
            print('memory of sentence_embed_list:', sys.getsizeof(sentence_embed_list),
                  'memory of sentence_embed_array:', sys.getsizeof(sentence_embed_array))
            del sentence_embed_list, word_embed_list, sentence_embed_array
            gc.collect()
            sentence_embed_list, word_embed_list = [], []
            sentence_embed_array = np.array(sentence_embed_list)
            print('memory of sentence_embed_list:', sys.getsizeof(sentence_embed_list),
                  'memory of sentence_embed_array:', sys.getsizeof(sentence_embed_array))
            breakpoint()

    oov_rate = oov_num / all_num
    print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)


# ****************************** QUERYING PPDB SERVICE ***********************************

''' Returns list of PPDB representatives '''


def queryPPDB(ppdb_url, phr_list):
    try:
        data = {"data": phr_list}
        headers = {'Content-Type': 'application/json'}
        req = requests.post(ppdb_url + 'ppdbAll', data=json.dumps(data), headers=headers)

        if (req.status_code == 200):
            data = json.loads(req.text)
            return data['data']
        else:
            print("Error! Status code :" + str(req.status_code))

    except Exception as e:
        print("Error in getGlove service!! \n\n", e)


def getPPDBclusters(ppdb_url, phr_list, phr2id):
    ppdb_map = dict()
    raw_phr_list = [phr.split('|')[0] for phr in phr_list]
    rep_list = queryPPDB(ppdb_url, raw_phr_list)

    for i in range(len(phr_list)):
        if rep_list[i] == None: continue  # If no representative for phr then skip

        phrId = phr2id[phr_list[i]]
        ppdb_map[phrId] = rep_list[i]

    return ppdb_map


def getPPDBclustersRaw(ppdb_url, phr_list):
    ppdb_map = dict()
    raw_phr_list = [phr.split('|')[0] for phr in phr_list]
    rep_list = queryPPDB(ppdb_url, raw_phr_list)

    for i, phr in enumerate(phr_list):
        if rep_list[i] == None: continue  # If no representative for phr then skip
        ppdb_map[phr] = rep_list[i]

    return ppdb_map

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
# ***************************************** TEXT SPLIT ***********************************************
def proc_ent(ent):
    ent = ent.lower().replace('.', ' ').replace('-', ' ').strip().replace('_', ' ').replace('|', ' ').strip()
    # ent = ent.lower().replace('.', ' ').replace('-', ' ').strip().replace('_', ' ').replace('|', ' ').strip()
    #ent = ' '.join([tok.decode('utf-8').split('/')[0] for tok in wnl.lemmatize(ent)])
    #ent = ' '.join([tok.split('/')[0] for tok in wnl.lemmatize(ent)])
    ent = wnl.lemmatize(ent)
    # ent = ' '.join(list( set(ent.split()) - set(config.stpwords)))
    return ent


def wordnetDisamb(sent, wrd):
    res = lesk(sent, wrd)
    if len(dir(res)) == 92:
        return res.name()
    else:
        return None


def getLogger(name, log_dir, config_dir):
    config_dict = json.load(open(config_dir + '/log_config.json'))

    if os.path.isdir(log_dir) == False:  # Make log_dir if doesn't exist
        os.system('mkdir {}'.format(log_dir))

    config_dict['handlers']['file_handler']['filename'] = log_dir + '/' + name
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger
