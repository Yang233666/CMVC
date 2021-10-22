import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split

# part 2 - bert fine-tuned
import torch
from torch import nn
from torch import optim
from transformers import BertModel, BertTokenizer

class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('../data/bert-base-uncased')
        self.bert = BertModel.from_pretrained('../data/bert-base-uncased')
        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, batch_sentences):
        # a = batch_sentences[4]
        # print('a:', type(a), len(a), a)
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=60,
                                                           pad_to_max_length=True)  # tokenize、add special token、pad
        # print('batch_tokenized:', type(batch_tokenized), batch_tokenized)
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        # print('input_ids:', type(input_ids), input_ids)
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        # print('attention_mask:', type(attention_mask), attention_mask)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state)
        return linear_output

class BERT_Model(object):

    def __init__(self, train_set):
        self.train_set = train_set
        self.sentences = self.train_set[0].values
        self.targets = self.train_set[1].values
        self.batch_size = 512
        self.epochs = 20
        self.lr = 0.005
        self.print_every_batch = 5

    def fine_tune(self):
        # print('sentences:', type(self.sentences), len(self.sentences), self.sentences.shape)
        # print('targets:', type(self.targets), len(self.targets), self.targets.shape)
        # sentences: <class 'numpy.ndarray'> 3000 (3000,) a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films
        # targets: <class 'numpy.ndarray'> 3000 (3000,) [1 0 0 1 1 1 0 1 0 0]
        self.train_inputs, self.test_inputs, self.train_targets, self.test_targets = train_test_split(self.sentences, self.targets)
        print('train_inputs:', type(self.train_inputs), len(self.train_inputs), self.train_inputs.shape, self.train_inputs[0])
        print('test_inputs:', type(self.test_inputs), len(self.test_inputs), self.test_inputs.shape, self.test_inputs[0])
        print('train_targets:', type(self.train_targets), len(self.train_targets), self.train_targets.shape, self.train_targets[0:10])
        print('test_targets:', type(self.test_targets), len(self.test_targets), self.test_targets.shape, self.test_targets[0:10])
        # train_inputs: <class 'numpy.ndarray'> 2250 (2250,) stumbles over every cheap trick in the book trying to make the outrage come even easier
        # test_inputs: <class 'numpy.ndarray'> 750 (750,) may lack the pungent bite of its title , but it 's an enjoyable trifle nonetheless
        # train_targets: <class 'numpy.ndarray'> 2250 (2250,) [0 1 0 1 0 0 1 1 0 0]
        # test_targets: <class 'numpy.ndarray'> 750 (750,) [1 1 1 0 1 0 1 1 1 0]

        batch_count = int(len(self.train_inputs) / self.batch_size)
        batch_train_inputs, batch_train_targets = [], []
        for i in range(batch_count):
            batch_train_inputs.append(self.train_inputs[i * self.batch_size: (i + 1) * self.batch_size])
            batch_train_targets.append(self.train_targets[i * self.batch_size: (i + 1) * self.batch_size])

        # train the model
        bert_classifier_model = BertClassificationModel()
        optimizer = optim.SGD(bert_classifier_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            print_avg_loss = 0
            for i in range(batch_count):
                inputs = batch_train_inputs[i]
                labels = torch.tensor(batch_train_targets[i])
                optimizer.zero_grad()
                output = bert_classifier_model(inputs)
                # print('outputs:', type(outputs), outputs.shape, outputs)
                # print('labels:', type(labels), labels.shape, labels)
                # outputs: <class 'torch.Tensor'> torch.Size([512, 750])
                # labels: <class 'torch.Tensor'> torch.Size([512])
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                print_avg_loss += loss.item()
                if i % self.print_every_batch == (self.print_every_batch - 1):
                    print("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / self.print_every_batch))
                    print_avg_loss = 0

            # eval the trained model
            total = len(self.test_inputs)
            hit = 0
            with torch.no_grad():
                for i in range(total):
                    outputs = bert_classifier_model([self.test_inputs[i]])
                    _, predicted = torch.max(outputs, 1)
                    if predicted == self.test_targets[i]:
                        hit += 1
            real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
            print(real_time, 'epoch:', epoch, 'all epochs', self.epochs, "Accuracy: %.2f%%" % (hit / total * 100))


# train_df = pd.read_csv('../data/SST_train.csv', delimiter='\t', header=None)
# train_set = train_df[:3000]   #取其中的3000条数据作为我们的数据集
# print("Train set shape:", train_set.shape)
# BM = BERT_Model(train_set)
# BM.fine_tune()

sentence_list = ['Hello word!', 'Meltsinyourmouth', 'StarbucksFreeWi-Fi', 'WX-Subway', 'DQ1940']
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('../data/bert-base-uncased')
for sentence in sentence_list:
    word_list = tokenizer.tokenize(sentence)
    print('分词前：', sentence)
    print('分词后:', word_list)
    print()