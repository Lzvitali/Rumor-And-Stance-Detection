from model.models import GRUMultiTask
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import model.defines as df

# Preprocessed data paths
preprocessed_data_paths = {
    'test_rumors_tweets path':          '..\\data\\preprocessed data\\test\\rumors_tweets.npy',
    'test_rumors_labels path':          '..\\data\\preprocessed data\\test\\rumors_labels.npy',
    'test_stances_tweets path':         '..\\data\\preprocessed data\\test\\stances_tweets.npy',
    'test_stances_labels path':         '..\\data\\preprocessed data\\test\\stances_labels.npy',
}

batch_size_test_rumors = 28
batch_size_test_stances = 533


def run_model(model, device, test_loader_specific, criterion_specific, output_dim):
    test_losses = []
    num_correct = 0
    h_val = model.init_hidden()
    h_prev, _, h_prev_shared = tuple([e.data for e in h_val])

    model.eval()
    for inputs, labels in test_loader_specific:
        inputs, labels = inputs.to(device), labels.to(device)

        if output_dim == df.output_dim_rumors:
            output, h_prev_shared, h_prev = model(inputs, h_prev_shared, df.task_rumors_no,
                                                  h_prev_rumors=h_prev)
        else:
            output, h_prev_shared, h_prev = model(inputs, h_prev_shared, df.task_stances_no,
                                                  h_prev_stances=h_prev)

        # test_loss = criterion_specific(output, (torch.max(labels, 1)[1]).to(device))  # with CrossEntropyLoss
        test_loss = criterion_specific(output, labels.float())  # with BCELoss
        test_losses.append(test_loss.item())

        for out, label in zip(output, labels):
            max_idx = torch.argmax(out, dim=0)  # dim=0 means: look in the first row
            one_hot = torch.nn.functional.one_hot(torch.tensor([max_idx]), output_dim)
            one_hot = torch.squeeze(one_hot, 0)
            one_hot = one_hot.to(device)
            if torch.equal(label, one_hot):
                num_correct += 1

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_loader_specific.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))


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
    model_gru_rumors = GRUMultiTask(df.input_length, df.hidden_length)
    model_gru_stances = GRUMultiTask(df.input_length, df.hidden_length)
    model_gru_rumors.to(device)
    model_gru_stances.to(device)

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()  # with CrossEntropyLoss
    criterion = nn.BCELoss()  # with BCELoss

    # Loading the best model of rumor detection
    model_gru_rumors.load_state_dict(torch.load('./state_dict_rumors.pt'))

    # Loading the best model of stance classification
    model_gru_stances.load_state_dict(torch.load('./state_dict_stances.pt'))

    print('For Rumors:')
    run_model(model_gru_rumors, device, test_loader_rumors, criterion, df.output_dim_rumors)
    print('-----------------------------------------')
    print('For Stances:')
    run_model(model_gru_stances, device, test_loader_stances, criterion, df.output_dim_stances)


if __name__ == '__main__':
    main()
