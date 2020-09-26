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

batch_size_training_rumors = 75  # 56  # 149  # 28
batch_size_training_stances = 1130  # 900  # 2260  # 450

batch_size_validation_rumors = 14  # 28  # 14
batch_size_validation_stances = 525  # 1049  # 525

lr = 0.001  # 0.005  # learning rate
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
    # criterion = nn.CrossEntropyLoss()  # with CrossEntropyLoss
    # criterion = nn.L1Loss() # with L1Loss
    criterion = nn.MSELoss()  # with MSELoss
    # criterion = nn.BCELoss()  # with BCELoss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train the model
    model.train()  # set the model to train mode

    counter_batches = 0
    valid_loss_min_rumors = np.Inf
    valid_loss_min_stances = np.Inf

    for i in range(epochs):
        # h_prev_task_rumors = torch.zeros(df.hidden_length).to(device)
        # h_prev_task_stances = torch.zeros(df.hidden_length).to(device)
        # h_prev_shared = torch.zeros(df.hidden_length).to(device)

        h = model.init_hidden()

        for (inputs_rumors, labels_rumors), (inputs_stances, labels_stances) \
                in zip(train_loader_rumors, train_loader_stances):
            counter_batches += 1

            # -------------------------- Training ----------------------------------
            # make the training for rumors
            h_prev_task_rumors, h_prev_task_stances, h_prev_shared = tuple([e.data for e in h])
            inputs_rumors, labels_rumors = inputs_rumors.to(device), labels_rumors.to(device)
            optimizer.zero_grad()
            output_r, h_prev_shared, h_prev_task_rumors = model(inputs_rumors, h_prev_shared, df.task_rumors_no,
                                                                h_prev_rumors=h_prev_task_rumors)
            # loss_rumors = criterion(output_r, (torch.max(labels_rumors, 1)[1]).to(device))  # with CrossEntropyLoss
            # loss_rumors = criterion(output_r, labels_rumors)  # with L1Loss
            loss_rumors = criterion(output_r, labels_rumors.float())  # with BCELoss OR  with MSELoss
            print('\nRumors loss: ' + str(loss_rumors.item()))
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
            loss_rumors.backward()  # retain_graph=True
            optimizer.step()
            # End: Training -----------------------------------------------------------

            # -------------------------- Validation ----------------------------------
            # make the validation for rumors
            print('Validation for rumor detection model: ')
            val_losses = []
            h_val = model.init_hidden()
            h_prev_task_rumors_val, h_prev_task_stances_val, h_prev_shared_val = tuple([e.data for e in h_val])
            model.eval()
            for inp, lab in val_loader_rumors:
                inp, lab = inp.to(device), lab.to(device)
                out_v_r, h_prev_shared_val, h_prev_task_rumors_val = model(inp, h_prev_shared_val, df.task_rumors_no,
                                                                           h_prev_rumors=h_prev_task_rumors_val)
                # val_loss = criterion(out_v_r, (torch.max(lab, 1)[1]).to(device))  # with CrossEntropyLoss
                # val_loss = criterion(out_v_r, lab)  # with L1Loss
                val_loss = criterion(out_v_r, lab.float())  # with BCELoss OR  with MSELoss
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(i + 1, epochs),
                  "batch: {}...".format(counter_batches),
                  "Loss: {:.6f}...".format(loss_rumors.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min_rumors:
                torch.save(model.state_dict(), './state_dict_rumors.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min_rumors,
                                                                                                np.mean(val_losses)))
                valid_loss_min_rumors = np.mean(val_losses)
            # End: Validation -----------------------------------------------------------

            # -------------------------- Training ----------------------------------
            # make the training for stances
            h_prev_task_rumors, h_prev_task_stances, h_prev_shared = tuple([e.data for e in h])
            inputs_stances, labels_stances = inputs_stances.to(device), labels_stances.to(device)
            optimizer.zero_grad()
            output_s, h_prev_shared, h_prev_task_stances = model(inputs_stances, h_prev_shared, df.task_stances_no,
                                                                 h_prev_stances=h_prev_task_stances)
            # loss_stances = criterion(output_s, (torch.max(labels_stances, 1)[1]).to(device))  # with CrossEntropyLoss
            # loss_stances = criterion(output_s, labels_stances)  # with L1Loss
            loss_stances = criterion(output_s, labels_stances.float())  # with BCELoss OR  with MSELoss
            print('\nStances loss: ' + str(loss_stances.item()))
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
            loss_stances.backward()
            optimizer.step()
            # End: Training -----------------------------------------------------------

            # -------------------------- Validation ----------------------------------
            # make the validation for stances
            print('Validation for stance classification model: ')
            val_losses = []
            h_val = model.init_hidden()
            h_prev_task_rumors_val, h_prev_task_stances_val, h_prev_shared_val = tuple([e.data for e in h_val])
            model.eval()
            for inp, lab in val_loader_stances:
                inp, lab = inp.to(device), lab.to(device)
                out_v_s, h_prev_shared_val, h_prev_task_stances_val = model(inp, h_prev_shared_val, df.task_stances_no,
                                                                            h_prev_stances=h_prev_task_stances_val)
                # val_loss = criterion(out_v_s, (torch.max(lab, 1)[1]).to(device))  # with CrossEntropyLoss
                # val_loss = criterion(out_v_s, lab)  # with L1Loss
                val_loss = criterion(out_v_s, lab.float())  # with BCELoss OR  with MSELoss
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(i + 1, epochs),
                  "batch: {}...".format(counter_batches),
                  "Loss: {:.6f}...".format(loss_stances.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min_stances:
                torch.save(model.state_dict(), './state_dict_stances.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min_stances,
                                                                                                np.mean(val_losses)))
                valid_loss_min_stances = np.mean(val_losses)
            # End: Validation -----------------------------------------------------------


if __name__ == '__main__':
    main()

