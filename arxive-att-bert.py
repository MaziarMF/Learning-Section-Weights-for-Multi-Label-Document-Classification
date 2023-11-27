#!/usr/bin/env python
# coding: utf-8



import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
import sys
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split




# loading the arxive dataset
def load_arxive_ds(DATA_PATH = "../data-arxive/"):
    df = pd.read_csv(DATA_PATH, sep='\t')
    train_df, valid_df = train_test_split(df, test_size=0.10, random_state=42)
    test_df = pd.read_csv(DATA_PATH, sep='\t')
    return train_df, test_df, valid_df

train_df, test_df, valid_df = load_arxive_ds('./data/Arxive_data_AbsTitle.tsv')




train_df.drop('Unnamed: 0', axis=1, inplace=True)
test_df.drop('Unnamed: 0', axis=1, inplace=True)
valid_df.drop('Unnamed: 0', axis=1, inplace=True)
train_df.rename(columns={'Title': 'title', 'Abastract': 'abstract'}, inplace=True)
test_df.rename(columns={'Title': 'title', 'Abastract': 'abstract'}, inplace=True)
valid_df.rename(columns={'Title': 'title', 'Abastract': 'abstract'}, inplace=True)




target_list = [col for col in train_df.columns if col not in ['title', 'abstract']]




# hyperparameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05




#defining tokenizer
pretrained_model_name = "allenai/scibert_scivocab_cased"
#pretrained_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)




train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)




# creating a class for custom dataset
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, tokenizer, max_len):
        
        self.tokenizer = tokenizer
        self.df = df
        self.max_len = max_len
        self.title = self.df['title']
        self.abstract = self.df['abstract']
        self.targets = self.df[target_list].values
    
    def __len__(self):
        return len(self.title)
    
    def __getitem__(self, index):
        
        title = str(self.title[index])
        title = " ".join(title.split())
        
        abstract = str(self.abstract[index])
        abstract = " ".join(abstract.split())
        
        
        inputs_title = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens = True,
            padding = 'max_length',
            max_length=512,
            return_token_type_ids = True,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
            
        )
        
        inputs_abstract = self.tokenizer.encode_plus(
            abstract,
            None,
            add_special_tokens = True,
            padding = 'max_length',
            max_length=512,
            return_token_type_ids = True,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
            
        )
        
        
        return {
            'input_ids_title': inputs_title['input_ids'].flatten(),
            'attention_mask_title': inputs_title['attention_mask'].flatten(),
            'token_type_ids_title': inputs_title['token_type_ids'].flatten(),
            'input_ids_abstract': inputs_abstract['input_ids'].flatten(),
            'attention_mask_abstract': inputs_abstract['attention_mask'].flatten(),
            'token_type_ids_abstract': inputs_abstract['token_type_ids'].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }




train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset = CustomDataset(valid_df, tokenizer, MAX_LEN)




# defining train loaders
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = TRAIN_BATCH_SIZE,
    num_workers = 0,
    shuffle = True
)

val_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size = VALID_BATCH_SIZE,
    shuffle = False,
    num_workers = 0
)




# defining device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




def save_ckp(state, is_best, checkpoint_path, best_model_path):
    
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    torch.save(state, f_path)
    
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_path, model, optimizer):
    
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()




# crearing Bert Class
class BertClass(torch.nn.Module):
    
    def __init__(self):
        
        super(BertClass, self).__init__()
        dropout_rate = 0.3
        bert_size = 768
        nr_classes = len(target_list)
        att_layer_1 = 256
        att_layer_2 = 1
        pre_output = 256
        self.bert_model = BertModel.from_pretrained(pretrained_model_name, return_dict=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.att_l1 = torch.nn.Linear(bert_size, att_layer_1)
        self.att_l2 = torch.nn.Linear(att_layer_1, att_layer_2)
        self.softmax_layer = torch.nn.Softmax(dim=1)
        self.prefinal_layer = torch.nn.Linear(bert_size, pre_output)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.linear = torch.nn.Linear(pre_output, len(target_list))
    
    def forward(self, input_id_title, attn_mask_title, token_type_ids_title, 
               input_id_abstract, attn_mask_abstract, token_type_ids_abstract):
        
        
        output_title = self.bert_model(
            input_id_title, 
            attention_mask = attn_mask_title, 
            token_type_ids = token_type_ids_title
        )
        
        output_abstract = self.bert_model(
            input_id_abstract, 
            attention_mask = attn_mask_abstract, 
            token_type_ids = token_type_ids_abstract
        )
        
        #getting bert output for each input
        output_dropout_title = self.dropout(output_title.pooler_output)
        output_dropout_abstract = self.dropout(output_abstract.pooler_output)
        
        output_dropout_title = self.dropout(output_title.pooler_output)
        output_dropout_abstract = self.dropout(output_abstract.pooler_output)
        
        #first attention later
        attention_title_1 = self.relu(self.att_l1(output_dropout_title))
        attention_abstract_1 = self.relu(self.att_l1(output_dropout_abstract))
        
        attention_title_2 = self.relu(self.att_l2(attention_title_1))
        attention_abstract_2 = self.relu(self.att_l2(attention_abstract_1))
        
        # list of outputs of attention model
        list_att = [attention_abstract_2, attention_title_2]


        # applying softmax function
        torch_list = torch.cat((attention_abstract_2, attention_title_2), 1)
        att_scalars = self.softmax_layer(torch_list)
        

        # multiplying attention layers by obtained scalars
        embeddings_abstract = att_scalars[:, 0] * torch.transpose(output_dropout_abstract, 0, 1)
        embeddings_title = att_scalars[:, 1] * torch.transpose(output_dropout_title, 0, 1)

        # computing the final embedding
        final_embedding = embeddings_abstract + embeddings_title 
        final_embedding = torch.transpose(final_embedding, 0, 1)
        # adding another layer which maps the final embedding to the output
        prefinal_output = self.prefinal_layer(final_embedding)
        output = self.linear(prefinal_output)
        return output, att_scalars

model = BertClass()
model.to(device)




# defining loss function
def loss_fn(outputs, targets):
    
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

#defining the optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)




val_targets=[]
val_outputs=[]




# training the model

def train_model(n_epochs, training_loader, validation_loader, model, 
                optimizer, checkpoint_path, best_model_path):
    
  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf
   
 
  for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        ids_abstract = data['input_ids_abstract'].to(device, dtype = torch.long)
        mask_abstract = data['attention_mask_abstract'].to(device, dtype = torch.long)
        token_type_ids_abstract = data['token_type_ids_abstract'].to(device, dtype = torch.long)
        ids_title = data['input_ids_title'].to(device, dtype = torch.long)
        mask_title = data['attention_mask_title'].to(device, dtype = torch.long)
        token_type_ids_title = data['token_type_ids_title'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs, _ = model(ids_title, mask_title, 
                        token_type_ids_title, ids_abstract, mask_abstract, token_type_ids_abstract)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
    
    print('############# Epoch {}: Training End     #############'.format(epoch))
    torch.save(model.state_dict(), './model/pt_arxive_firsttry_{}'.format(epoch))
    
    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################    
    # validate the model #
    ######################
 
    model.eval()
   
    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):            
            
            ids_abstract = data['input_ids_abstract'].to(device, dtype = torch.long)
            mask_abstract = data['attention_mask_abstract'].to(device, dtype = torch.long)
            token_type_ids_abstract = data['token_type_ids_abstract'].to(device, dtype = torch.long)
            ids_title = data['input_ids_title'].to(device, dtype = torch.long)
            mask_title = data['attention_mask_title'].to(device, dtype = torch.long)
            token_type_ids_title = data['token_type_ids_title'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs, _ = model(ids_title, mask_title, 
                            token_type_ids_title, ids_abstract, mask_abstract, token_type_ids_abstract)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics 
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
      
      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }
        
        # save checkpoint

      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model




ckpt_path = "./curr_ckpt"
best_model_path = "./best_model.pt"




trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, ckpt_path, best_model_path)



DEVICE = torch.device('cuda:0')
model = BertClass()
model.to(DEVICE) 
model.load_state_dict(torch.load('./model/pt_arxive_firsttry_10'))
# model.eval()
list_output = []
list_attention_weights = []

for i in range(len(test_df)):
    title = test_df['title'][i]
    abstract = test_df['abstract'][i]
    
    inputs_title = tokenizer.encode_plus(
        title,
        None,
        add_special_tokens = True,
        padding = 'max_length',
        max_length=512,
        return_token_type_ids = True,
        truncation = True,
        return_attention_mask = True,
        return_tensors = 'pt'

    )
    inputs_abstract = tokenizer.encode_plus(
            abstract,
            None,
            add_special_tokens = True,
            padding = 'max_length',
            max_length=512,
            return_token_type_ids = True,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
            
        )
        
        
    model.eval()
    with torch.no_grad():
        
        ids_abstract = inputs_abstract['input_ids'].to(device, dtype = torch.long)
        mask_abstract = inputs_abstract['attention_mask'].to(device, dtype = torch.long)
        token_type_ids_abstract = inputs_abstract['token_type_ids'].to(device, dtype = torch.long)
        ids_title = inputs_title['input_ids'].to(device, dtype = torch.long)
        mask_title = inputs_title['attention_mask'].to(device, dtype = torch.long)
        token_type_ids_title = inputs_title['token_type_ids'].to(device, dtype = torch.long)
        outputs, att_weights = model(ids_title, mask_title, 
                            token_type_ids_title, ids_abstract, mask_abstract, token_type_ids_abstract)        
        final_output = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
    list_output.append(final_output)
    list_attention_weights.append(att_weights)
        #print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])




arr_pred = np.reshape(np.asarray(list_output), (np.asarray(list_output).shape[0], np.asarray(list_output).shape[2]))
pred_test = test_df[target_list].values
y_pred = (arr_pred > 0.5) 



from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
print('macro precision is:', precision_score(y_true=pred_test, y_pred=y_pred, average='macro'))
print('macro recall is:', recall_score(y_true=pred_test, y_pred=y_pred, average='macro'))
print('macro f1 is:', f1_score(y_true=pred_test, y_pred=y_pred, average='macro'))
print('micro precision is:', precision_score(y_true=pred_test, y_pred=y_pred, average='micro'))
print('micro recall is:', recall_score(y_true=pred_test, y_pred=y_pred, average='micro'))
print('micro f1 is:', f1_score(y_true=pred_test, y_pred=y_pred, average='micro'))
print('accuracy is:', accuracy_score(y_true=pred_test, y_pred=y_pred))
print('weighted precision is:', precision_score(y_true=pred_test, y_pred=y_pred, average='weighted'))
print('weighted recall is:', recall_score(y_true=pred_test, y_pred=y_pred, average='weighted'))
print('weighted f1 is:', f1_score(y_true=pred_test, y_pred=y_pred, average='weighted'))





