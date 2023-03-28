# dependencies
from datasets import load_dataset
from transformers import AutoModel
import torchvision.transforms as transforms
from torchvision.io import read_image
import numpy as np
from datasets import Features, ClassLabel, Array3D, Image
import torch
from torch import nn, tensor
import os
from torchmetrics import Accuracy, MatthewsCorrCoef
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import ViltProcessor, ViltModel
from PIL import Image
import requests
from tqdm import tqdm
from transformers import Trainer
import math
import pickle
import PIL
import transformers



os.environ["HF_ENDPOINT"] = "https://huggingface.co"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################# data ###################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_PATH="/home/benjamin/Desktop/ml/neuMultiModal/sarcasm/"
TEXT_LENGTH=75
TEXT_HIDDEN=256

def load_image_labels():
    # get labels
    img2labels=dict()
    with open(os.path.join(WORKING_PATH,"extract/","extract_all.txt"),"rb") as file:
        for line in file:
            content=eval(line)
            img2labels[int(content[0])]=content[1:]
    # label to index of embedding, dict, word:value 0~1001
    label2index=pickle.load(open(os.path.join(WORKING_PATH,"ExtractWords/vocab.pickle"), 'rb'))
    return img2labels,label2index

img2labels,label2index=load_image_labels()

def attribute_indices(id, label2index):
        labels=img2labels[id]
        label_index=list(map(lambda label:label2index[label],labels))
        return torch.tensor(label_index)

# skeleton
def load_data(img2labels):
    data_set=dict()
    data_set['train'] = []
    data_set['test'] = []
    data_set['valid'] = []
    # separate into train, test, and validation categories
    for dataset in ["train"]:
        file=open(os.path.join(WORKING_PATH,"text/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[2]
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                path_image = WORKING_PATH + 'image_data/' + image+ '.jpg'
                data_set['train'].append({"text":sentence,"label":group, "attributes":img2labels[int(image)]})
    for dataset in ["test"]:
        file=open(os.path.join(WORKING_PATH,"text/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[3] #2
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set['test'].append({"text":sentence,"label":group, "attributes":img2labels[int(image)]})
    for dataset in ["valid"]:
        file=open(os.path.join(WORKING_PATH,"text/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[3] #2
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set['valid'].append({"text":sentence,"label":group, "attributes":img2labels[int(image)]})
    return data_set

dataset = load_data(img2labels)


"""
Here is the code for adding a classification layer to the VilBert model
"""
class classificationBERT(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.execution = nn.Sequential(nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 2), nn.Softmax(dim=1))

    
    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        # the last hidden state or the pooled outputs
        pooled_output = outputs[1]
        logits = self.execution(pooled_output)
        return logits

class CustomTrainer(Trainer):
    def __init__(self, epochs, lr, test_data, train_data,  model, processor, num_classes):
        self.epochs = epochs
        self.lr = lr
        self.model = model
        self.processor = processor
        self.num_classes = num_classes
        self.test_data = test_data
        self.train_data = train_data
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.BCELoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def plot(self, loss, epoch):
        timesteps = np.arange(1, loss.shape[0] + 1)
        # Plot the MSE vs timesteps
        plt.plot(timesteps, loss)
        # Add axis labels and a title
        plt.xlabel('Timestep')
        plt.ylabel('BCE Loss')
        plt.title('Loss')
        plt.savefig('./loss/loss_' + str(epoch) + '.png')
        plt.close()

    def train(self, batch_size):
        adam = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(adam, self.epochs)
        loss_fct = nn.BCELoss()
        accuracy = Accuracy(task='multiclass', num_classes=2).to(device)
        mcc = MatthewsCorrCoef(task='binary').to(device)
        counter = 0

        training_loss_over_epochs = []

        # so lets make a baseline with the sarcasm dataset for this model
        # then a text only baseline
        for epoch in range(self.epochs):
            accuracy = Accuracy("binary").to(device)
            training_loss = []
            total_acc = 0
            num_train_steps = math.floor(len(self.train_data)/batch_size) * batch_size
            runtime = 0
            value = 0
            for train_index in tqdm(range(0, num_train_steps, batch_size)):
                self.model.zero_grad()
                text = [d['text'] for d in self.train_data[train_index:train_index+batch_size]]
                try:
                    inputs = self.processor(text, padding = True, return_tensors="pt").to(device)
                except ValueError:
                    value += 1
                    continue
                try:
                    out = self.model(**inputs)
                except RuntimeError:
                    print('runtime')
                    runtime += 1
                    continue
                truth = torch.nn.functional.one_hot(torch.tensor([d['label'] for d in self.train_data[train_index:train_index+batch_size]]), num_classes=self.num_classes)
                loss = loss_fct(out.float().to(device), truth.view(batch_size, 2).float().to(device))
                training_loss.append(loss.item())
                maximums = torch.argmax(out, dim = 1)
                truth_max = torch.argmax(truth, dim = 1)
                accuracy.update(maximums.to(device), truth_max.to(device))
                adam.zero_grad()
                loss.backward()
                adam.step()

            acc = accuracy.compute()
            print("Accuracy:", acc)
            print('runtime errors: ', runtime)
            print('value errors: ', value)
            #self.plot(np.array(training_loss), epoch)
            print('\n')
            print('epoch: ', counter)
            counter += 1
            print('loss total: ', sum(training_loss))
            print('\n')
            training_loss_over_epochs.append(training_loss)
            #exponential.step()
            cosine.step()
            self.model.eval()

        with torch.no_grad():
            num_test_steps = len(self.test_data)
            total_acc = 0
            runtime = 0
            value = 0
            accuracy_two = Accuracy("binary").to(device)
            for y in tqdm(range(num_test_steps)):
                text = self.test_data[y]['text']
                try:
                    inputs = self.processor(text, return_tensors="pt").to(device)
                except ValueError:
                    value += 1
                    continue
                try:
                    out = self.model(**inputs)
                except RuntimeError:
                    runtime += 1
                    continue
                truth = torch.nn.functional.one_hot(torch.tensor(self.test_data[y]['label']), num_classes=self.num_classes)
                maximums = torch.tensor([torch.argmax(out).item()])
                truth_max = torch.tensor([torch.argmax(truth).item()])
                accuracy_two.update(maximums.to(device), truth_max.to(device))
                total_acc += (maximums == truth_max)
        accuracy = total_acc / y
        print('runtime errors: ', runtime)
        print('value errors: ', value)
        print("Basic accuracy on test set: ", accuracy_two.compute())
        return training_loss_over_epochs, accuracy


processor = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = transformers.BertModel.from_pretrained('bert-base-uncased')
model =  classificationBERT(bert).to(device)
#text = dataset['train'][0]['text']
#inputs = processor(text, return_tensors="pt").to(device)
#print(inputs)
#print(model.forward(**inputs))
train = CustomTrainer(5, 1e-4, dataset['test'], dataset['train'], model, processor, 2)
# hyperparameter search?
#train.train(32)  

