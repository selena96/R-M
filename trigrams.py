import pandas as pd
import numpy as np
import sys

import torch
import torch.nn as nn


def read_data(file_name): # this function reads data and adds ## in front and end of each verb
    cols=['present_phon','past_phon']
    data = pd.read_csv(file_name, sep="\t", usecols=cols)

    data['present_phon']="##"+data['present_phon']+"##"
    data['past_phon']="##"+data['past_phon']+"##"

    return data


def all_phonemes(data): #this function returns list of all phonemes
    phonemes=set()

    for verb in data['present_phon']:
        for i in range(len(verb)):
            phonemes.add(verb[i])

    for verb in data['past_phon']:
        for i in range(len(verb)):
            phonemes.add(verb[i])
    
    list_of_phonemes=list(sorted(phonemes))

    return list_of_phonemes


def all_trigrams_to_file(data, list_of_phonemes): # this function just makes the list of all possible trigrams
    all_trigrams=[]

    with open("alltrigrams.csv", "w") as file:

        for i in range(0,len(list_of_phonemes)-1):
            for j in range(0,len(list_of_phonemes)-1):
                for k in range(0,len(list_of_phonemes)-1):
                    trigram = list_of_phonemes[i]+list_of_phonemes[j]+list_of_phonemes[k]
                    all_trigrams.append(trigram)
                    file.write(trigram)
                    file.write('\n')

    file.close()

    return all_trigrams


def make_trigrams(verb): # given verb, this function makes all trigrams
    trigrams = []

    for i in range(len(verb)-2):
        trigrams.append(verb[i]+verb[i+1]+verb[i+2])
        
    return trigrams


def verb_to_vec(verb): # given verb, this function makes true/false vector of trigrams
    verb_trigrams = make_trigrams(verb)
    sorted(verb_trigrams)
    vect = []
    curr = 0

    for trigram in all_trigrams:
        if (verb_trigrams[curr]==trigram):
            vect.append(1)
            curr+=1
        else:
            vect.append(0)
    assert(len(all_trigrams)==len(vect))

    return vect

data = read_data("CELEXmod.csv")
list_of_phonemes = all_phonemes(data)
all_trigrams = all_trigrams_to_file(data, list_of_phonemes)


X = np.array(data['present_phon'].tolist())
Y = np.array(data['past_phon'].tolist())

X1=[] 
Y1=[]
for i in range(0,len(X)):
    X1.append(verb_to_vec(X[i]))
    Y1.append(verb_to_vec(Y[i]))

print("Preprocessing finished")

class NgramModel(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.out = nn.Linear(embedding_size, vocab_size)
        
        # Not sure whether I need next line
        #self.out.weight = self.embedding.weight

    def forward(self, x): # <- This function actually runs the model
        emb = self.embedding(x)

        # Sigmoid is built in criterion so it is not needed here
        logits = self.out(emb)

        return logits

losses = [] # later we will print all losses

inputs = torch.tensor(X1)
targets = torch.tensor(Y1)

dataset = torch.utils.data.TensorDataset(inputs, targets) 
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True) # batches and shuffles data

model = NgramModel(len(X1), 300)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(3):

    running_loss = 0.0
    for i, data in enumerate(loader):

        inputs, labels = data

        optimizer.zero_grad()

        # run model
        log_probs = model(inputs)

        pos_weight = torch.ones([64])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(log_probs, targets)

        loss.backward() 
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        
        losses.append(running_loss)
        
print('Finished Training')
print('Losses: ')
print(losses)