from model.models import GRUMultiTask
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import model.defines as df


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
}

batch_size_training_rumors = 56  # 149  # 28
batch_size_training_stances = 900  # 2260  # 450

batch_size_validation_rumors = 14  # 28  # 14
batch_size_validation_stances = 525  # 1049  # 525

lr = 0.005  # 0.005  # learning rate
epochs = 10


def main():
    # for rumors
    train_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['training_rumors_tweets path'])),
                                      torch.from_numpy(np.load(preprocessed_data_paths['training_rumors_labels path'])))
    val_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['validation_rumors_tweets path'])),
                                    torch.from_numpy(np.load(preprocessed_data_paths['validation_rumors_labels path'])))

    train_loader_rumors = DataLoader(train_data_rumors, shuffle=False, batch_size=batch_size_training_rumors)
    val_loader_rumors = DataLoader(val_data_rumors, shuffle=False, batch_size=batch_size_validation_rumors)

    # for stances
    train_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['training_stances_tweets path'])),
                                       torch.from_numpy(np.load(preprocessed_data_paths['training_stances_labels path'])))
    val_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['validation_stances_tweets path'])),
                                     torch.from_numpy(np.load(preprocessed_data_paths['validation_stances_labels path'])))

    train_loader_stances = DataLoader(train_data_stances, shuffle=False, batch_size=batch_size_training_stances)
    val_loader_stances = DataLoader(val_data_stances, shuffle=False, batch_size=batch_size_validation_stances)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # if we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # create the model
    model = GRUMultiTask(df.input_length, df.hidden_length)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train the model
    model.train()  # set the model to train mode

    counter_batches = 0
    valid_loss_min_rumors = np.Inf
    valid_loss_min_stances = np.Inf

    for i in range(epochs):
        h_prev_task_rumors = torch.zeros(df.hidden_length).to(device)
        h_prev_task_stances = torch.zeros(df.hidden_length).to(device)
        h_prev_shared = torch.zeros(df.hidden_length).to(device)

        for (inputs_rumors, labels_rumors), (inputs_stances, labels_stances) \
                in zip(train_loader_rumors, train_loader_stances):
            counter_batches += 1

            # for rumors
            inputs_rumors, labels_rumors = inputs_rumors.to(device), labels_rumors.to(device)
            optimizer.zero_grad()
            output_r, h_prev_shared, h_prev_task_rumors = model(inputs_rumors, h_prev_shared, df.task_rumors_no,
                                                                h_prev_rumors=h_prev_task_rumors)

            loss_rumors = criterion(output_r, (torch.max(labels_rumors, 1)[1]).to(device))
            print('Rumors loss: ' + str(loss_rumors.item()))
            loss_rumors.backward(retain_graph=True)
            optimizer.step()

            # for stances
            inputs_stances, labels_stances = inputs_stances.to(device), labels_stances.to(device)
            optimizer.zero_grad()
            output_s, h_prev_shared, h_prev_task_stances = model(inputs_stances, h_prev_shared, df.task_stances_no,
                                                                 h_prev_stances=h_prev_task_stances)

            loss_stances = criterion(output_s, (torch.max(labels_stances, 1)[1]).to(device))
            print('Stances loss: ' + str(loss_stances.item()))
            loss_stances.backward()
            optimizer.step()


if __name__ == '__main__':
    main()

