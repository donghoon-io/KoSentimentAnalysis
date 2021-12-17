
# coding: utf-8
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install transformers')



import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, notebook
from tokenization_kobert import KoBertTokenizer
from transformers import AdamW
from transformers import BertModel, DistilBertModel
from transformers.optimization import get_cosine_schedule_with_warmup


device = torch.device("cpu") # initiate GPU
bertmodel = BertModel.from_pretrained('monologg/kobert')


df_train = pd.read_excel('train.xlsx')
df_validatation = pd.read_excel('validation.xlsx')


df_train = df_train[df_train['감정_대분류'].str.contains('분노|슬픔|불안|당황|상처|기쁨')]
df_train['감정_대분류'] = df_train['감정_대분류'].replace(['분노', '슬픔', '불안', '당황', '상처', '기쁨', '분노 ', '슬픔 ', '불안 ', '당황 ', '상처 ', '기쁨 '], [0,1,2,3,4,5,0,1,2,3,4,5])
df_train.head()

df_train_data = []

for q, label in zip(df_train['사람문장1'], df_train['감정_대분류'])  :
    data = []
    data.append(q)
    data.append(str(label))

    df_train_data.append(data)



df_validatation = df_validatation[df_validatation['감정_대분류'].str.contains('분노|슬픔|불안|당황|상처|기쁨')]
df_validatation['감정_대분류'] = df_validatation['감정_대분류'].replace(['분노', '슬픔', '불안', '당황', '상처', '기쁨', '분노 ', '슬픔 ', '불안 ', '당황 ', '상처 ', '기쁨 '], [0,1,2,3,4,5,0,1,2,3,4,5])
df_validatation.head()

df_validation_data = []

for q, label in zip(df_validatation['사람문장1'], df_validatation['감정_대분류'])  :
    data = []
    data.append(q)
    data.append(str(label))

    df_validation_data.append(data)

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        self.sentences = [(np.array(tokenizer.encode(i[sent_idx], max_length=64, pad_to_max_length=True)).astype(np.int32),
                           np.array(len(tokenizer.encode(i[sent_idx], max_length=64, pad_to_max_length=False))).astype(np.int32),
                           np.zeros(shape=(64,), dtype=np.int32)) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
        
        print(self.sentences[0])
        
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class_num = 6 # set the number of classes

max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

data_train = BERTDataset(df_train_data, 0, 1, tokenizer, max_len, True, False)
data_test = BERTDataset(df_validation_data, 0, 1, tokenizer, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size = 768, num_classes = 6, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate) # Adam optimizer
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


# training

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(notebook.tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
    model.eval()
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(notebook.tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
        
    print("epoch {} validation acc {}".format(e+1, test_acc / (batch_id+1)))


# test

new_test = nlp.data.TSVDataset('test.tsv', field_indices=[1,2], num_discard_samples=1)
test_set = BERTDataset(new_test , 0, 1, tok, max_len, True, False)
test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)

for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_input)): 
  token_ids = token_ids.long().to(device) 
  segment_ids = segment_ids.long().to(device) 
  valid_length= valid_length 
  out = model(token_ids, valid_length, segment_ids)
  prediction = out.cpu().detach().numpy().argmax()
  
  print(batch_id + "번째 문장 분류 예측값: " + prediction)
