"""
Lets train (with alternate tweet structure)
"""

import torch
from torch import tensor, nn, autograd
import matplotlib.pyplot as plt
from michinaga.src import teanet
from tqdm import tqdm
import numpy as np
from random_data_alt import random_data
from torchmetrics import Accuracy, MatthewsCorrCoef

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def train(model, params):

    """
    A simple training loop. The aim is to train teanet over a specified number of epochs, and to measure accuracy with a basic measure.

    args: 
        - model
            the teanet model to be trained

        - params 
            the parameters to construct the training process
    """

    x_train_tweets = params['x_tweet_train']
    x_price_train = params['x_price_train']
    y_train = params['y_train']

    x_test_tweets = params['x_tweet_test']
    x_test_price = params['x_price_test']
    y_test = params['y_test']

    epochs = params['epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']

    lag = params['lag']
    model.to(device)
    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # learning rate schedulers, so that the saddle point doesn't debilitate model performance

    exponential = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=0.95)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(adam, epochs)

    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    training_loss_over_epochs = []
    accuracy = Accuracy(task='multiclass', num_classes=2).to(device)
    mcc = MatthewsCorrCoef(task='binary').to(device)


    """
    TRAIN
    """
    # we should measure the training accuracy
    for e in tqdm(range(epochs)):

        train_index = 0
        training_loss = []
        total_acc = 0
        total_mc = 0

        while(train_index < len(x_train_tweets) - batch_size):
            model.zero_grad()
            # when training on gpu ensure that the device is set correctly
            # a chunk of an np array cannot be set onto a device?
            #x_input = [x_train_tweets[train_index:train_index+batch_size].to(device), x_price_train[train_index:train_index+batch_size].view(batch_size, lag, 4).to(device)]
            x_input = [x_train_tweets[train_index:train_index+batch_size], x_price_train[train_index:train_index+batch_size].view(batch_size, lag, 4)]     
            out = model.forward(x_input)
            loss = loss_fn(out.view(batch_size, 2).float(), y_train[train_index:train_index+batch_size].float().to(device))
            training_loss.append(loss.item())
            #print(loss.item())
            # accuracy for training data
            maximums = torch.max(out.view(batch_size, 2), dim = 1).indices
            max_targets = torch.max(y_train[train_index:train_index+batch_size], dim = 1).indices

            # here is the accuracy measurement
            acc = accuracy(maximums.float().to(device), max_targets.float().to(device))
            mc = mcc(maximums.float().to(device), max_targets.float().to(device))
            total_mc += mc
            total_acc += (acc * batch_size)
            # for debugging
            #with autograd.detect_anomaly():
            adam.zero_grad()
            loss.backward()
            adam.step()

            train_index += batch_size
        print('\n')
        print('epoch: ', e)
        print('training set accuracy: ', total_acc/train_index)
        # the average matthews correlation coefficient?
        print('matthews correlation coefficient', total_mc/train_index)
        # the total matthews correlation coefficient
        print('total matthews correlation coefficient', total_mc)
        print('loss total: ', sum(training_loss))
        print('\n')
        training_loss_over_epochs.append(training_loss)
        #exponential.step()
        cosine.step()

    torch.save(model, 'trained_teanet.pt')
    
    """
    EVALUATE

    For the first trial, I just want to see the basic accuracy. The more acute measurements can be implemented later. 
    """
    #model = torch.load('trained_teanet.pt')
    model.eval()

    with torch.no_grad():
        num_correct = 0
        tot = 0
        model.setBatchSize(1)
        actuals = []
        outputs = []
        for y in tqdm(range(int(y_test.shape[0]))):
            out = model.forward([x_test_tweets[y].to(device), x_test_price[y].view(1, lag, 4).to(device)])
            actual = torch.max(y_test[y].to(device), dim = 0).indices
            out_index = torch.max(out, dim = 2).indices
            if(actual.item() == out_index.item()):
                num_correct += 1
            tot += 1
            # mcc coefficient from these list
            actuals.append(actual.item())
            outputs.append(out_index.item())
    act = torch.tensor(actuals).float().to(device)
    outs = torch.tensor(outputs).float().to(device)
    mat = mcc(outs, act)
    accuracy = num_correct / tot
    print("Basic accuracy on test set: ", accuracy)
    print("mcc: ", mat)
    return training_loss_over_epochs, accuracy

def plot(arr_list, legend_list, color_list, ylabel, fig_title):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis

        Note that, make sure the elements in the arr_list, legend_list and color_list are associated with each other correctly.
        Do not forget to change the ylabel for different plots.
    """
    # set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps")

    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err *= 1.96
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3,
                        color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)

    plt.show()

if __name__ == "__main__":
    lag = 5
    batch_size = 5
    randomize = random_data()
    accuracy_over_time = []
    model = teanet(5, 100, 2, batch_size, lag, 100, 50)
    #model = torch.load('trained_teanet.pt')

    """
    This call randomizes the data, so that each epoch set is trained on a different set of train and test
    data
    """

   # randomize.forward()

    # change this to handle the new types of data storage
    
    params = {
        'x_tweet_train': np.load('x_train_tweets.pt.npy', allow_pickle=True),
        'x_price_train': torch.load('x_train_prices.pt'), 
        'y_train': torch.load('y_train.pt'),
        'x_tweet_test': np.load('x_test_tweets.pt.npy', allow_pickle=True),
        'x_price_test': torch.load('x_test_prices.pt'),
        'y_test': torch.load('y_test.pt'),
        'lag': lag,
        'batch_size': batch_size,
        'epochs': 1,
        'learning_rate': 1e-3
    }

    training_loss, accuracy = train(model, params)
    accuracy_over_time.append(accuracy)


