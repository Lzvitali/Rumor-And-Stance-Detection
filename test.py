from model.models import GRUMultiTask
import os
import train as tr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import defines as df

# Preprocessed data paths
preprocessed_data_paths = {
    'test_rumors_tweets path':      os.path.join('data', 'preprocessed data', 'test', 'rumors_tweets.npy'),
    'test_rumors_labels path':      os.path.join('data', 'preprocessed data', 'test', 'rumors_labels.npy'),
    'test_stances_tweets path':     os.path.join('data', 'preprocessed data', 'test', 'stances_tweets.npy'),
    'test_stances_labels path':     os.path.join('data', 'preprocessed data', 'test', 'stances_labels.npy'),
}

batch_size_test_rumors = 20  # 200
batch_size_test_stances = 91  # 918


def main():
    # for rumors
    test_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['test_rumors_tweets path'])),
                                     torch.from_numpy(np.load(preprocessed_data_paths['test_rumors_labels path'])))
    test_loader_rumors = DataLoader(test_data_rumors, shuffle=False, batch_size=batch_size_test_rumors, drop_last=True)

    # for stances
    test_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['test_stances_tweets path'])),
                                      torch.from_numpy(np.load(preprocessed_data_paths['test_stances_labels path'])))
    test_loader_stances = DataLoader(test_data_stances, shuffle=False, batch_size=batch_size_test_stances, drop_last=True)

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
                                    loss_func='BCELoss',
                                    is_dropout=False
                                    )
    model_multi_task.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()

    # Loading the model
    model_multi_task.load_state_dict(torch.load('model/model_state_dict.pt'))

    # Run the model
    cnt_correct_test_rumors, cnt_correct_test_stances = tr.validation_or_testing(model_multi_task, test_loader_rumors,
                                                                                 test_loader_stances, criterion, device,
                                                                                 operation='testing')
    print('-----------------------------------------\n')
    validation_acc_rumors = cnt_correct_test_rumors / ((len(test_loader_rumors.dataset) / batch_size_test_rumors)
                                                       * batch_size_test_rumors)
    print('Test accuracy rumors: {:.3f}%'.format(validation_acc_rumors * 100))

    validation_acc_stances = cnt_correct_test_stances / ((len(test_loader_stances.dataset) / batch_size_test_stances)
                                                         * batch_size_test_stances)
    print('Test accuracy stances: {:.3f}%'.format(validation_acc_stances * 100))


if __name__ == '__main__':
    main()
