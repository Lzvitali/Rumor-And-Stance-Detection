from model.models import GRUTaskSpecific, GRUCELLShared
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

    # create models
    model_shared = GRUCELLShared(df.input_length, df.hidden_length)
    model_gru_rumors = GRUTaskSpecific(model_shared, df.output_dim_rumors, df.task_rumors_no, df.input_length, df.hidden_length)
    model_gru_stances = GRUTaskSpecific(model_shared, df.output_dim_stances, df.task_stances_no, df.input_length, df.hidden_length)
    model_shared.to(device)
    model_gru_rumors.to(device)
    model_gru_stances.to(device)

    # for rumors
    criterion_rumors = nn.CrossEntropyLoss()
    optimizer_rumors = torch.optim.Adam(model_gru_rumors.parameters(), lr=lr)

    # for stances
    criterion_stances = nn.CrossEntropyLoss()
    optimizer_stances = torch.optim.Adam(model_gru_stances.parameters(), lr=lr)

    model_gru_rumors.train()
    model_gru_stances.train()

    counter_batches = 0
    valid_loss_min_rumors = np.Inf
    valid_loss_min_stances = np.Inf

    for i in range(epochs):
        h_prev_shared_rumors = model_shared.init_state().to(device)
        h_prev_task_rumors = model_gru_rumors.init_state().to(device)
        h_prev_shared_stances = model_shared.init_state().to(device)
        h_prev_task_stances = model_gru_stances.init_state().to(device)

        for (inputs_rumors, labels_rumors), (inputs_stances, labels_stances) \
                in zip(train_loader_rumors, train_loader_stances):
            counter_batches += 1

            # for rumors
            inputs_rumors, labels_rumors = inputs_rumors.to(device), labels_rumors.to(device)

            optimizer_rumors.zero_grad()
            out_r, h_prev_shared_rumors, h_prev_task_rumors = model_gru_rumors(inputs_rumors, h_prev_shared_rumors,
                                                                               h_prev_task_rumors)
            out_r = out_r.to(device)
            loss_rumors = criterion_rumors(out_r, (torch.max(labels_rumors, 1)[1]).to(device))  # (torch.max(labels_rumors, 1)[1]).to(device)
            print('Rumors loss: ' + str(loss_rumors.item()))
            h_prev_shared_rumors.detach_()
            h_prev_task_rumors.detach_()
            loss_rumors.backward()
            optimizer_rumors.step()

            # for stances
            inputs_stances, labels_stances = inputs_stances.to(device), labels_stances.to(device)

            optimizer_stances.zero_grad()
            out_s, h_prev_shared_stances, h_prev_task_stances = model_gru_stances(inputs_stances,h_prev_shared_stances,
                                                                                  h_prev_task_stances)
            out_s = out_s.to(device)
            loss_stances = criterion_stances(out_s, (torch.max(labels_stances, 1)[1]).to(device))  # (torch.max(labels_stances, 1)[1]).to(device)
            print('Stances loss: ' + str(loss_stances.item()))
            h_prev_shared_stances.detach_()
            h_prev_task_stances.detach_()
            loss_stances.backward()
            optimizer_stances.step()

            # make the validation for rumors
            print('\nValidation for rumor detection model: ')
            val_losses = []
            h_t_prev_shared_validation = model_shared.init_state().to(device)
            h_t_prev_task_validation = model_gru_rumors.init_state().to(device)
            model_gru_rumors.eval()
            for inp, lab in val_loader_rumors:
                inp, lab = inp.to(device), lab.to(device)
                out_v_r, h_t_prev_shared_validation, h_t_prev_task_validation = model_gru_rumors(inp,
                                                                                                 h_t_prev_shared_validation,
                                                                                                 h_t_prev_task_validation)
                out_v_r = out_v_r.to(device)
                val_loss = criterion_rumors(out_v_r, (torch.max(lab, 1)[1]).to(device))  # (torch.max(lab, 1)[1]).to(device)
                val_losses.append(val_loss.item())

            model_gru_rumors.train()
            print("Epoch: {}/{}...".format(i + 1, epochs),
                  "batch: {}...".format(counter_batches),
                  "Loss: {:.6f}...".format(loss_rumors.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min_rumors:
                torch.save(model_gru_rumors.state_dict(), './state_dict_rumors.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min_rumors,
                                                                                                np.mean(val_losses)))
                valid_loss_min_rumors = np.mean(val_losses)

            # make the validation for stances
            print('\nValidation for stance classification model: ')
            val_losses = []
            h_t_prev_shared_validation = model_shared.init_state().to(device)
            h_t_prev_task_validation = model_gru_stances.init_state().to(device)
            model_gru_stances.eval()
            for inp, lab in val_loader_stances:
                inp, lab = inp.to(device), lab.to(device)
                out_v_s, h_t_prev_shared_validation, h_t_prev_task_validation = model_gru_stances(inp,
                                                                                              h_t_prev_shared_validation
                                                                                              , h_t_prev_task_validation)
                val_loss = criterion_stances(out_v_s, (torch.max(lab, 1)[1]).to(device))  # (torch.max(lab, 1)[1]).to(device)
                val_losses.append(val_loss.item())

            model_gru_stances.train()
            print("Epoch: {}/{}...".format(i + 1, epochs),
                  "batch: {}...".format(counter_batches),
                  "Loss: {:.6f}...".format(loss_stances.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min_stances:
                torch.save(model_gru_stances.state_dict(), './state_dict_stances.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min_stances,
                                                                                                np.mean(val_losses)))
                valid_loss_min_stances = np.mean(val_losses)


if __name__ == '__main__':
    main()

