from model.models import GRUMultiTask
import os
import train as tr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import defines as df

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle


# Preprocessed data paths
preprocessed_data_paths = {
    'test_rumors_tweets path':      os.path.join('data', 'preprocessed data', 'test', 'rumors_tweets.npy'),
    'test_rumors_labels path':      os.path.join('data', 'preprocessed data', 'test', 'rumors_labels.npy'),
    'test_stances_tweets path':     os.path.join('data', 'preprocessed data', 'test', 'stances_tweets.npy'),
    'test_stances_labels path':     os.path.join('data', 'preprocessed data', 'test', 'stances_labels.npy'),
}

batch_size_test_rumors = 4
batch_size_test_stances = 11

# 2  4   8    16   32   64    100
# 5  11  22   44   88   176   274

loss_function = 'BCELoss'      # supported options: CrossEntropyLoss | BCELoss | L1Loss | MSELoss


def main():
    # for rumors
    test_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['test_rumors_tweets path'])),
                                     torch.from_numpy(np.load(preprocessed_data_paths['test_rumors_labels path'])))
    test_loader_rumors = DataLoader(test_data_rumors, shuffle=False, batch_size=batch_size_test_rumors, drop_last=False)

    # for stances
    test_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['test_stances_tweets path'])),
                                      torch.from_numpy(np.load(preprocessed_data_paths['test_stances_labels path'])))
    test_loader_stances = DataLoader(test_data_stances, shuffle=False, batch_size=batch_size_test_stances, drop_last=False)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # if we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # create the model
    model_multi_task = GRUMultiTask(input_length=df.input_length,
                                    hidden_length_rumors=df.hidden_length_rumors,
                                    hidden_length_stances=df.hidden_length_stances,
                                    hidden_length_shared=df.hidden_length_shared,
                                    loss_func=loss_function,
                                    is_dropout=False
                                    )
    model_multi_task.to(device)

    # Loss
    if loss_function == 'BCELoss':
        criterion = nn.BCELoss()
    elif loss_function == 'L1Loss':
        criterion = nn.L1Loss()
    elif loss_function == 'MSELoss':
        criterion = nn.MSELoss()
    else:  # the default
        criterion = nn.CrossEntropyLoss()

    # Loading the model
    model_multi_task.load_state_dict(torch.load(os.path.join('model', 'model_state_dict.pt')))

    # Load hidden states
    try:
        with open(os.path.join('model', 'h_prevs.pickle'), 'rb') as fp:
            h_training = pickle.load(fp)
            h_training = (torch.from_numpy(h_training['h_1']).to(device),
                          torch.from_numpy(h_training['h_2']).to(device),
                          torch.from_numpy(h_training['h_3']).to(device))
    except EnvironmentError:
        h_training = model_multi_task.init_hidden()

    # Run the model
    accuracy_r, accuracy_s = tr.validation_or_testing(model_multi_task, test_loader_rumors,
                                                      test_loader_stances, criterion, device,
                                                      h_training, operation='testing')
    print('-----------------------------------------\n')
    print('Test accuracy rumors: {:.3f}%'.format(accuracy_r))
    print('Test accuracy stances: {:.3f}%'.format(accuracy_s))


if __name__ == '__main__':
    main()
