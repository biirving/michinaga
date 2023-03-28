import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import PIL
import pickle
import torch
from PIL import Image
from wordEmbedding import wordEmbedding
from tqdm import tqdm


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
    data_set['train'] = {}
    data_set['test'] = {}
    data_set['valid'] = {}
    # separate into train, test, and validation categories
    for dataset in ["train"]:
        file=open(os.path.join(WORKING_PATH,"text/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[2]
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set['train'][int(image)]={"text":sentence,"label":group, "attributes":img2labels[int(image)]}
    for dataset in ["test"]:
        file=open(os.path.join(WORKING_PATH,"text/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[3] #2
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set['test'][int(image)]={"text":sentence,"label":group, "attributes":img2labels[int(image)]}
    for dataset in ["valid"]:
        file=open(os.path.join(WORKING_PATH,"text/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[3] #2
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set['valid'][int(image)]={"text":sentence,"label":group, "attributes":img2labels[int(image)]}
    return data_set

dataset = load_data(img2labels)

# 224 is a common input size for vision transformers
def image_loader(id):
    path= WORKING_PATH + 'image_data/' + str(id) + '.jpg'
    img_pil =  PIL.Image.open(path)
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    img_tensor = transform(img_pil)
    return img_tensor

#test_id = list(dataset['train'].keys())[0]
#test_text = dataset['train'][test_id]['text']
#test_label = dataset['train'][test_id]['label']
#test_attributes = dataset['train'][test_id]['attributes']
#test_img = image_loader(test_id)

word_embeds = wordEmbedding('twitter', 'average', False)

train_text = None
train_images = None
train_attributes = None
train_labels = None
train_keys = list(dataset['train'].keys())
# embed all of the tweets
for l in tqdm(range(len(train_keys))):
    id = train_keys[l]
    text = dataset['train'][id]['text']
    text_embeddings = word_embeds.embed(text)
    label = torch.nn.functional.one_hot(torch.tensor([dataset['train'][id]['label']]), num_classes= 2).to(device)
    attributes = dataset['train'][id]['attributes']
    attribute_embeddings = word_embeds.embed(attributes).to(device)
    image = image_loader(id).to(device)
    train_text = text_embeddings.view((1, text_embeddings.shape[0])).to(device)
    train_images = image.view((1, 3, 224, 224)).to(device)
    train_labels = label.view((1, 2)).to(device)
    train_attributes = attribute_embeddings.view((1, attribute_embeddings.shape[0])).to(device)
    torch.save(train_text, WORKING_PATH + 'train/text/train_text_' + str(l) + '.pt')
    torch.save(train_images, WORKING_PATH + 'train/images/train_images_' + str(l) + '.pt')
    torch.save(train_labels, WORKING_PATH + 'train/labels/train_labels_' + str(l) +  '.pt')
    torch.save(train_attributes, WORKING_PATH + 'train/attributes/train_attributes_' + str(l) + '.pt')

test_text = None
test_images = None
test_attributes = None
test_labels = None

test_keys = list(dataset['test'].keys())
# embed all of the tweets
for l in tqdm(range(len(test_keys))):
    id = test_keys[l]
    text = dataset['test'][id]['text']
    text_embeddings = word_embeds.embed(text)
    label = torch.nn.functional.one_hot(torch.tensor([dataset['test'][id]['label']]), num_classes= 2).to(device)
    attributes = dataset['test'][id]['attributes']
    attribute_embeddings = word_embeds.embed(attributes).to(device)
    image = image_loader(id).to(device)
    test_text = text_embeddings.view((1, text_embeddings.shape[0])).to(device)
    test_images = image.view((1, 3, 224, 224)).to(device)
    test_labels = label.view((1, 2)).to(device)
    test_attributes = attribute_embeddings.view((1, attribute_embeddings.shape[0])).to(device)
    torch.save(test_text, WORKING_PATH + 'test/text/test_text_'+ str(l) + '.pt')
    torch.save(test_images, WORKING_PATH + 'test/images/test_images_'+ str(l) + '.pt')
    torch.save(test_labels, WORKING_PATH + 'test/labels/test_labels_'+ str(l) + '.pt')
    torch.save(test_attributes, WORKING_PATH + 'test/attributes/test_attributes_'+ str(l) + '.pt')

valid_text = None
valid_keys = list(dataset['valid'].keys())
# embed all of the tweets
for l in tqdm(range(len(valid_keys))):
    id = valid_keys[l]
    text = dataset['valid'][id]['text']
    text_embeddings = word_embeds.embed(text)
    label = torch.nn.functional.one_hot(torch.tensor([dataset['valid'][id]['label']]), num_classes= 2).to(device)
    attributes = dataset['valid'][id]['attributes']
    attribute_embeddings = word_embeds.embed(attributes).to(device)
    image = image_loader(id).to(device)
    valid_text = text_embeddings.view((1, text_embeddings.shape[0])).to(device)
    valid_images = image.view((1, 3, 224, 224)).to(device)
    valid_labels = label.view((1, 2)).to(device)
    valid_attributes = attribute_embeddings.view((1, attribute_embeddings.shape[0])).to(device)
    torch.save(valid_text, WORKING_PATH + 'valid/text/valid_text_'+ str(l) + '.pt')
    torch.save(valid_images, WORKING_PATH + 'valid/images/valid_images_'+ str(l) + '.pt')
    torch.save(valid_labels, WORKING_PATH + 'valid/labels/valid_labels_'+ str(l) + '.pt')
    torch.save(valid_attributes, WORKING_PATH + 'valid/attributes/valid_attributes_'+ str(l) + '.pt')