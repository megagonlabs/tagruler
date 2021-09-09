import pandas as pd
import numpy as np
from datareader import TextDatasetReader
import pickle
import torch

# for bert
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize

# for elmo
from allennlp.modules.elmo import Elmo, batch_to_ids

# for sbert
from sentence_transformers import SentenceTransformer

# for split
from sklearn.model_selection import train_test_split

# define paths
data_path = "dataset.csv"

# load data
df = pd.read_csv(data_path)

# create .nlp file
reader = TextDatasetReader()
data = reader.read(data_path)
with open('processed.nlp','wb') as f:
    pickle.dump(data,f)

# create .bert file
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True) # Whether the model returns all hidden-states.)
max_len = 128
all_embs = []
with torch.no_grad():
    for i,d in enumerate(data):
        sent = [x.text.lower() for x in d['tokens']]
        tokens = ["[CLS]"] + sent + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(token_ids) + [0] * (max_len-len(token_ids))
        token_ids = token_ids + [0] * (max_len-len(token_ids))
        temp_output = model(torch.LongTensor([token_ids]),torch.LongTensor([masks]))
        all_embs.append(temp_output[2][12][0][1:1+len(sent)])
        print(i,sent)
bert_embs = np.array([normalize(t.detach().numpy()) for t in all_embs])

with open('processed.bert','wb') as f:
    np.save(f,bert_embs)

# create .elmo file

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 1, dropout=0)
all_embs = []
for i,d in enumerate(data):
    sent = [[x.text for x in d['tokens']]]
    character_ids = batch_to_ids(sent)
    emb = elmo(character_ids)
    all_embs.append(emb['elmo_representations'][0][0,:,:])
    print(i,' '.join(sent[0]))

elmo_embs = np.array([normalize(t.detach().numpy()) for t in all_embs])
with open('processed.elmo','wb') as f:
    np.save(f,elmo_embs)


# create .sbert file

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
sberts = model.encode(df['text'])
sberts_normal = np.array([normalize(s.reshape(1,-1)).reshape(-1) for s in sberts])
with open('processed.sbert','wb') as f:
    np.save(f,sberts_normal)

# split and create .csv file
i_train,i_test = train_test_split(np.array(list(range(len(df)))), test_size=0.5, random_state=0)
i_test,i_dev = train_test_split(i_test, test_size=0.6, random_state=0)
i_dev,i_valid = train_test_split(i_dev, test_size=0.33, random_state=0)

split = np.array(["_"]*len(df),dtype='object')
split[i_train]='train'
split[i_test]='test'
split[i_dev]='dev'
split[i_valid]='valid'
df['split']=split
df.to_csv('processed.csv') 