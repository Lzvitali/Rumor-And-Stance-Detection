from model.models import GRUMultiTask
import train as tr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import defines as df

# Preprocessed data paths
preprocessed_data_paths = {
    'test_rumors_tweets path':  'data\\preprocessed data\\test\\rumors_tweets.npy',
    'test_rumors_labels path':  'data\\preprocessed data\\test\\rumors_labels.npy',
    'test_stances_tweets path': 'data\\preprocessed data\\test\\stances_tweets.npy',
    'test_stances_labels path': 'data\\preprocessed data\\test\\stances_labels.npy',
}

batch_size_test_rumors = 56
batch_size_test_stances = 1066


def main():
    # for rumors
    test_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['test_rumors_tweets path'])),
                                     torch.from_numpy(np.load(preprocessed_data_paths['test_rumors_labels path'])))
    test_loader_rumors = DataLoader(test_data_rumors, shuffle=False, batch_size=batch_size_test_rumors)

    # for stances
    test_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['test_stances_tweets path'])),
                                      torch.from_numpy(np.load(preprocessed_data_paths['test_stances_labels path'])))
    test_loader_stances = DataLoader(test_data_stances, shuffle=False, batch_size=batch_size_test_stances)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # if we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # create the model
    model_gru_rumors = GRUMultiTask(input_length=df.input_length, hidden_length=df.hidden_length, loss_func='CrossEntropyLoss')
    model_gru_stances = GRUMultiTask(input_length=df.input_length, hidden_length=df.hidden_length, loss_func='CrossEntropyLoss')
    model_gru_rumors.to(device)
    model_gru_stances.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # with CrossEntropyLoss

    # Loading the best model of rumor detection
    model_gru_rumors.load_state_dict(torch.load('model/state_dict_rumors.pt'))

    # Loading the best model of stance classification
    model_gru_stances.load_state_dict(torch.load('model/state_dict_stances.pt'))

    # For Rumors
    cnt_correct_test_rumors = tr.validation_or_testing(model_gru_rumors, 'rumor', test_loader_rumors, criterion,
                                                       device, operation='testing')
    validation_acc_rumors = cnt_correct_test_rumors / len(test_loader_rumors.dataset)
    print("Test accuracy rumors: {:.3f}%".format(validation_acc_rumors * 100))

    print('-----------------------------------------')

    # For Stances
    cnt_correct_test_stances = tr.validation_or_testing(model_gru_stances, 'stance', test_loader_stances, criterion,
                                                        device, operation='testing')
    validation_acc_stances = cnt_correct_test_stances / len(test_loader_stances.dataset)
    print("Test accuracy stances: {:.3f}%".format(validation_acc_stances * 100))


if __name__ == '__main__':
    main()
