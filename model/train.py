from model.models import GRUTaskSpecific, GRUCELLShared
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Preprocessed data paths
preprocessed_data_paths = {
    'training_rumors_tweets path':      '..\\data\\preprocessed data\\training\\rumors_tweets.npy',
    'training_rumors_labels path':      '..\\data\\preprocessed data\\training\\rumors_labels.npy',
    'training_stances_tweets path':     '..\\data\\preprocessed data\\training\\stances_tweets.npy',
    'training_stances_labels path':     '..\\data\\preprocessed data\\training\\stances_labels.npy',

    'validation_rumors_tweets path':    '..\\data\\preprocessed data\\validation\\rumors_tweets.npy',
    'validation_rumors_labels path':    '..\\data\\preprocessed data\\validation\\rumors_labels.npy',
    'validation_stances_tweets path':   '..\\data\\preprocessed data\\validation\\stances_tweets.npy',
    'validation_stances_labels path':   '..\\data\\preprocessed data\\validation\\stances_labels.npy',

    'test_rumors_tweets path':          '..\\data\\preprocessed data\\test\\rumors_tweets.npy',
    'test_rumors_labels path':          '..\\data\\preprocessed data\\test\\rumors_labels.npy',
    'test_stances_tweets path':         '..\\data\\preprocessed data\\test\\stances_tweets.npy',
    'test_stances_labels path':         '..\\data\\preprocessed data\\test\\stances_labels.npy',
}

batch_size_training_rumors = 28
batch_size_training_stances = 450
batch_size_validation_rumors = 14
batch_size_validation_stances = 525
lr = 0.005
epochs = 3

# GRU params
input_length = 250
hidden_length = 100
output_dim_rumors = 3
output_dim_stances = 4


def main():
    # for rumors

    train_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['training_rumors_tweets path'])),
                                      torch.from_numpy(np.load(preprocessed_data_paths['training_rumors_labels path'])))
    val_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['validation_rumors_tweets path'])),
                                    torch.from_numpy(np.load(preprocessed_data_paths['validation_rumors_labels path'])))
    test_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['test_rumors_tweets path'])),
                                     torch.from_numpy(np.load(preprocessed_data_paths['test_rumors_labels path'])))

    train_loader_rumors = DataLoader(train_data_rumors, shuffle=False, batch_size=batch_size_training_rumors)
    val_loader_rumors = DataLoader(val_data_rumors, shuffle=False, batch_size=batch_size_validation_rumors)
    test_loader_rumors = DataLoader(test_data_rumors, shuffle=False, batch_size=batch_size_training_rumors)

    # for stances
    train_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['training_stances_tweets path'])),
                                       torch.from_numpy(np.load(preprocessed_data_paths['training_stances_labels path'])))
    val_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['validation_stances_tweets path'])),
                                     torch.from_numpy(np.load(preprocessed_data_paths['validation_stances_labels path'])))
    test_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['test_stances_tweets path'])),
                                      torch.from_numpy(np.load(preprocessed_data_paths['test_stances_labels path'])))

    train_loader_stances = DataLoader(train_data_stances, shuffle=False, batch_size=batch_size_training_stances)
    val_loader_stances = DataLoader(val_data_stances, shuffle=False, batch_size=batch_size_validation_stances)
    test_loader_stances = DataLoader(test_data_stances, shuffle=False, batch_size=batch_size_training_stances)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # if we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # create models
    model_shared = GRUCELLShared(input_length, hidden_length)
    model_gru_rumors = GRUTaskSpecific(model_shared, output_dim_rumors, input_length, hidden_length)
    model_gru_stances = GRUTaskSpecific(model_shared, output_dim_stances, input_length, hidden_length)
    model_shared.to(device)
    model_gru_rumors.to(device)
    model_gru_stances.to(device)

    # for rumors
    criterion_rumors = nn.BCELoss()
    optimizer_rumors = torch.optim.Adam(model_gru_rumors.parameters(), lr=lr)

    # for stances
    criterion_stances = nn.BCELoss()
    optimizer_stances = torch.optim.Adam(model_gru_stances.parameters(), lr=lr)

    model_gru_rumors.train()
    model_gru_stances.train()

    for i in range(epochs):
        for (inputs_rumors, labels_rumors), (inputs_stances, labels_stances) \
                in zip(train_loader_rumors, train_loader_stances):

            # for rumors
            inputs_rumors, labels_rumors = inputs_rumors.to(device), labels_rumors.to(device)

            def closure_rumors():
                optimizer_rumors.zero_grad()
                out = model_gru_rumors(inputs_rumors)
                loss = criterion_rumors(out, labels_rumors)
                print('loss:', loss.item())
                loss.backward()
                return loss

            optimizer_rumors.step(closure_rumors)

            # for rumors
            inputs_stances, labels_stances = inputs_stances.to(device), labels_stances.to(device)

            def closure_stances():
                optimizer_stances.zero_grad()
                out = model_gru_rumors(inputs_stances)
                loss = criterion_rumors(out, labels_stances)
                print('loss:', loss.item())
                loss.backward()
                return loss

            optimizer_stances.step(closure_stances)

            # make the validation
            # val_losses = []
            # model.eval()
            # for inp, lab in val_loader:
            #     val_h = tuple([each.data for each in val_h])
            #     inp, lab = inp.to(device), lab.to(device)
            #     out, val_h = model(inp, val_h)
            #     val_loss = criterion(out.squeeze(), lab.float())
            #     val_losses.append(val_loss.item())


if __name__ == '__main__':
    main()

