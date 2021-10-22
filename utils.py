import numpy as np
from numpy.linalg import norm
from gensim.models.word2vec import Word2Vec
import os
import time
#import tensorflow as tf
import json


def load_args(file_path):
    with open(file_path, 'r') as f:
        args_dict = json.load(f) # 将字符串转换为字典
        f.close()
    print("load arguments:", args_dict)
    args = ARGs(args_dict)
    return args


class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v) # 赋予属性值，这里的v就是(e,a,v)的v

'''
def load_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config) # tensorflow初始化配置时动态申请显存
'''

def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/') # 把每行的每个字符一个个分开，变成一个list. strip去掉字符串首尾的/，split拆分字符串，返回分割后的字符串list.
    path = params[-1] # [-1]表示将最后一块切割出来。str="http://www.runoob.com/python/att-string-split.html" print("0:%s"%str.split("/")[-1]) 结果:att-string-split.html
    folder = out_folder + method_name + '/' + path + "/" + div_path + str(time.strftime("%Y%m%d%H%M%S")) + "/" # 重新组成输出文件夹，time.strftime返回可读字符串表示的当地时间
    print("results output folder:", folder) # folder为输出文件夹
    return folder


def dict2file(file, dic): # 将字典转换为文件
    if dic is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in dic.items():
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()
    print(file, "saved.")


def save_embeddings(folder, kg, nv_ent_embeds, nv_rel_embeds, rv_ent_embeds, rv_rel_embeds):
    if not os.path.exists(folder): # 判断括号里的folder文件是否存在，括号内可以是文件路径
        os.makedirs(folder) # 在folder文件夹中递归创建目录
    if nv_ent_embeds is not None:
        np.save(folder + 'nv_ent_embeds.npy', nv_ent_embeds)
    if nv_rel_embeds is not None:
        np.save(folder + 'nv_rel_embeds.npy', nv_rel_embeds)
    if rv_ent_embeds is not None:
        np.save(folder + 'rv_ent_embeds.npy', rv_ent_embeds)
    if rv_rel_embeds is not None:
        np.save(folder + 'rel_embeds.npy', rv_rel_embeds)
    dict2file(folder + 'kg_ent_ids', kg.entities_id_dict)
    dict2file(folder + 'kg_rel_ids', kg.relations_id_dict)
    print("Embeddings saved!")


def read_word2vec0(file_path, vector_dimension=300):
    print('\n', file_path)
    word2vec = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split(' ')
            if len(line) != vector_dimension + 1:
                continue
            v = np.array(list(map(float, line[1:])), dtype=np.float32)
            word2vec[line[0]] = v
    file.close()
    return word2vec

  # wiki:line[1:len(line)-1], wiki_sub/crawl/crawl_sub/cc_en/glove:line[1:len(line)]
def read_word2vec(file_path, vector_dimension=300):
    print('\n', file_path)
    word2vec = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) > 100:
                # line = line.lower().strip('\n').split(' ')  # 把每行的每个字符一个个分开，变成一个list.
                line = line.strip('\n').split(' ')  # 把每行的每个字符一个个分开，变成一个list.
                # if len(line) != vector_dimension + 1:
                # continue
                v = np.array(list(map(float, line[1:len(line)])),
                             dtype=np.float32)  # map()接收函数float和一个list line，并通过把函数float依次作用在list的每个元素上，得到一个新的list并返回
                word2vec[line[0]] = v
    file.close()
    return word2vec

def lower_word2vec(word2vec): # 若key为唯一小写，则直接读入，若key为唯一大写，则改为小写读入，若key存在大写与小写，则均读入
    old_word2vec = word2vec
    print('old_word2vec', len(old_word2vec))
    word2vec_lower = dict()
    num1, num2 = 0, 0
    word2vec_lower = old_word2vec.copy()
    for keys, values in old_word2vec.items():
        key_lower = keys.lower()
        word2vec_lower.update({key_lower: values})
    print('word2vec_lower', len(word2vec_lower))
    print('lower in:', num1)
    print('lower not in:', num2)
    return word2vec_lower

def read_entity_local_name(file_path, entities_set): # 从文件夹路径读取实体名字
    entity_local_name = read_entity_local_name_file(file_path, entities_set)
    print("total local names:", len(entity_local_name))
    return entity_local_name

def read_relation_local_name(file_path, relations_set): # 从文件夹路径读取关系名字  # 新加的
    relation_local_name = read_relation_local_name_file(file_path, relations_set)
    print("total relation local names:", len(relation_local_name))
    return relation_local_name  # 新加的


def read_entity_local_name_file(file_path, entities_set): #从文件路径提取实体名字
    print('read entity local names from:', file_path)
    entity_local_name = dict()
    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file: # 遍历实体集中的每一行
            line = line.strip('\n').split('\t') # 把每行的每个字符一个个分开，变成一个list.
            assert len(line) == 2 # 断言每一行的长度为2，若不符合就报错
            if line[1] == '':
                cnt += 1
            ln = line[1]
            if ln.endswith(')'): # endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False。可选参数"start"与"end"为检索字符串的开始与结束位置。
                ln = ln.split('(')[0] # 将行中的括号内拆分出来
            entity_local_name[line[0]] = ln.replace('_', ' ')
    file.close()

    for e in entities_set:
        if e not in entity_local_name: # 若实体集中有实体与实体名字不符，则让这个实体名字为空
            entity_local_name[e] = ''
    print('len(entity_local_name)', len(entity_local_name))
    print('len(entities_set)', len(entities_set))
    #assert len(entity_local_name) == len(entities_set) # 断言实体名字的数量与实体集的数量相同
    return entity_local_name

def read_relation_local_name_file(file_path, relations_set): #从文件路径提取关系名字  # 新加的
    print('read relation local names from:', file_path)
    relation_local_name = dict()
    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file: # 遍历关系集中的每一行
            line = line.strip('\n').split('\t') # 把每行的每个字符一个个分开，变成一个list.
            assert len(line) == 2 # 断言每一行的长度为2，若不符合就报错
            if line[1] == '':
                cnt += 1
            ln = line[1]
            if ln.endswith(')'): # endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False。可选参数"start"与"end"为检索字符串的开始与结束位置。
                ln = ln.split('(')[0] # 将行中的括号内拆分出来
            relation_local_name[line[0]] = ln.replace('_', ' ')
    file.close()

    for e in relations_set:
        if e not in relation_local_name: # 若实体集中有实体与实体名字不符，则让这个实体名字为空
            relation_local_name[e] = ''
    print('len(relation_local_name)', len(relation_local_name))
    print('len(relations_set)', len(relations_set))
    #assert len(entity_local_name) == len(entities_set) # 断言实体名字的数量与实体集的数量相同
    return relation_local_name  # 新加的


def generate_word2vec_by_character_embedding(word_list, vector_dimension=300): # 给定单词列表利用word2vec生成300维的char_embed
    character_vectors = {}
    alphabet = ''
    ch_num = {}
    for word in word_list: # 对于单词列表中的每一个单词
        for ch in word: # 对于单词中的每一个字符
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n # 单词中的字符数
    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True) # 对ch_num中的元素按照第二个元素降序排列
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            alphabet += ch_num[i][0]
    #print(alphabet)
    #print('len(alphabet):', len(alphabet), '\n')
    char_sequences = [list(word) for word in word_list] # 提取输入的单词列表中的单词
    #print('char_sequences:', len(char_sequences), '\n')
    model = Word2Vec(char_sequences, size=vector_dimension, window=5, min_count=1)
    model.save('char_embeddings.vec')
    for ch in alphabet:
        assert ch in model
        character_vectors[ch] = model[ch] # 将word_vec生成的向量保存到char_embed

    word2vec = {}
    for word in word_list:
        vec = np.zeros(vector_dimension, dtype=np.float32) # 向量初始化 vec为类型为np的32位浮点数组成的300维0数组
        for ch in word:
            if ch in alphabet:
                vec += character_vectors[ch]
        if len(word) != 0:
            word2vec[word] = vec / len(word)
    return word2vec


def look_up_word2vec(id_tokens_dict, word2vec, tokens2vec_mode='add', keep_unlist=False, vector_dimension=300, tokens_max_len=5):
    if tokens2vec_mode == 'add':  # LP查找函数，若oi已经有word_embed,则直接将word_embed添加
        return tokens2vec_add(id_tokens_dict, word2vec, vector_dimension, keep_unlist)
    else: # 若LP查找函数发现oi没有现成的word_embed，则用tokens2vec_encoder生成一个向量
        return tokens2vec_encoder(id_tokens_dict, word2vec, vector_dimension, tokens_max_len, keep_unlist)


def tokens2vec_encoder(id_tokens_dict, word2vec, vector_dimension, tokens_max_len, keep_unlist):
    tokens_vectors_dict = {}
    for v_id, tokens in id_tokens_dict.items():
        words = tokens.split(' ')
        vectors = np.zeros((tokens_max_len, vector_dimension), dtype=np.float32) # 向量初始化
        flag = False
        for i in range(min(tokens_max_len, len(words))): # 设置tokens的最大数量为5
            if words[i] in word2vec:
                vectors[i] = word2vec[words[i]]
                flag = True
        if flag:
            tokens_vectors_dict[v_id] = vectors
    if keep_unlist:
        for v_id, _ in id_tokens_dict.items():
            if v_id not in tokens_vectors_dict:
                tokens_vectors_dict[v_id] = np.zeros((tokens_max_len, vector_dimension), dtype=np.float32)
    return tokens_vectors_dict


def tokens2vec_add(id_tokens_dict, word2vec, vector_dimension, keep_unlist):
    tokens_vectors_dict = {}
    cnt = 0
    for e_id, local_name in id_tokens_dict.items():
        words = local_name.split(' ')
        vec_sum = np.zeros(vector_dimension, dtype=np.float32)
        for word in words:
            if word in word2vec:
                vec_sum += word2vec[word]
        if sum(vec_sum) != 0:
            vec_sum = vec_sum / norm(vec_sum)
        elif not keep_unlist:
            cnt += 1
            continue
        tokens_vectors_dict[e_id] = vec_sum
    # print('clear_unlisted_value:', cnt)
    return tokens_vectors_dict


def look_up_char2vec(id_tokens_dict, character_vectors, vector_dimension=300):
    tokens_vectors_dict = {}
    for e_id, ln in id_tokens_dict.items():
        vec_sum = np.zeros(vector_dimension, dtype=np.float32)
        for ch in ln:
            if ch in character_vectors:
                vec_sum += character_vectors[ch]
        if sum(vec_sum) != 0:
            vec_sum = vec_sum / norm(vec_sum)
        tokens_vectors_dict[e_id] = vec_sum
    return tokens_vectors_dict


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def normalization(data):
    _range = np.max(abs(data))
    return data / _range

def normalization2(data):
    _range = np.sum(data)
    return data / _range

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_theta = 0.5 + 0.5 * cos_theta
    return cos_theta

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_distance = 1 - cos_theta
    # cos_distance = 0.5 - 0.5 * cos_theta
    return cos_distance