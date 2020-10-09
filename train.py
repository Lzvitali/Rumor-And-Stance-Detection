from model.models import GRUMultiTask
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import defines as df
from time import gmtime, strftime
from datetime import datetime
from sklearn.metrics import f1_score


# Preprocessed data paths
preprocessed_data_paths = {
    'training_rumors_tweets path':      os.path.join('data', 'preprocessed data', 'training', 'rumors_tweets.npy'),
    'training_rumors_labels path':      os.path.join('data', 'preprocessed data', 'training', 'rumors_labels.npy'),
    'training_stances_tweets path':     os.path.join('data', 'preprocessed data', 'training', 'stances_tweets.npy'),
    'training_stances_labels path':     os.path.join('data', 'preprocessed data', 'training', 'stances_labels.npy'),

    'validation_rumors_tweets path':    os.path.join('data', 'preprocessed data', 'validation', 'rumors_tweets.npy'),
    'validation_rumors_labels path':    os.path.join('data', 'preprocessed data', 'validation', 'rumors_labels.npy'),
    'validation_stances_tweets path':   os.path.join('data', 'preprocessed data', 'validation', 'stances_tweets.npy'),
    'validation_stances_labels path':   os.path.join('data', 'preprocessed data', 'validation', 'stances_labels.npy'),
}

batch_size_training_rumors = 58
batch_size_training_stances = 58

batch_size_validation_rumors = 20  # 200
batch_size_validation_stances = 96  # 961

loss_function = 'CrossEntropyLoss'      # supported options: CrossEntropyLoss | BCELoss | L1Loss | MSELoss
learning_rate = 0.0005                  # learning rate
epochs = 100

is_dropout = False  # can be True or False
drop_prob = 0.0


def main():
    # create 'TensorDataset's for rumors
    train_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['training_rumors_tweets path'])),
                                      torch.from_numpy(np.load(preprocessed_data_paths['training_rumors_labels path'])))
    val_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['validation_rumors_tweets path'])),
                                    torch.from_numpy(np.load(preprocessed_data_paths['validation_rumors_labels path'])))

    train_loader_rumors = DataLoader(train_data_rumors, shuffle=True, batch_size=batch_size_training_rumors, drop_last=True)
    val_loader_rumors = DataLoader(val_data_rumors, shuffle=False, batch_size=batch_size_validation_rumors, drop_last=True)

    # create 'TensorDataset's  for stances
    train_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['training_stances_tweets path'])),
                                       torch.from_numpy(np.load(preprocessed_data_paths['training_stances_labels path'])))

    val_data_stances = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['validation_stances_tweets path'])),
                                     torch.from_numpy(np.load(preprocessed_data_paths['validation_stances_labels path'])))

    # create 'DataLoader's  for stances
    train_loader_stances = DataLoader(train_data_stances, shuffle=True, batch_size=batch_size_training_stances, drop_last=True)
    val_loader_stances = DataLoader(val_data_stances, shuffle=False, batch_size=batch_size_validation_stances, drop_last=True)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # if we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # create the model
    model = GRUMultiTask(input_length=df.input_length,
                         hidden_length_rumors=df.hidden_length_rumors,
                         hidden_length_stances=df.hidden_length_stances,
                         hidden_length_shared=df.hidden_length_shared,
                         loss_func=loss_function,
                         is_dropout=is_dropout,
                         drop_prob=drop_prob
                         )
    model.to(device)

    # Loss
    if loss_function == 'BCELoss':
        criterion = nn.BCELoss()
    elif loss_function == 'L1Loss':
        criterion = nn.L1Loss()
    elif loss_function == 'MSELoss':
        criterion = nn.MSELoss()
    else:  # the default
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    model.train()  # set the model to train mode

    validation_min_loss = {
        'min loss': np.Inf
    }

    last_save = {
        'last save': 0,
    }

    # check time before training
    start_time = gmtime()
    start_time = strftime("%H:%M:%S", start_time)

    for i in range(epochs):
        h = model.init_hidden()
        print('\nEpoch: ' + str(i + 1))

        counter_batches = 0
        cnt_correct_training_rumors = 0
        cnt_correct_training_stances = 0
        cnt_correct_validation_rumors = 0
        cnt_correct_validation_stances = 0
        sum_loss_training_rumors = 0
        sum_loss_training_stances = 0

        # iterate through all the batch
        for (inputs_rumors, labels_rumors), (inputs_stances, labels_stances) \
                in zip(train_loader_rumors, train_loader_stances):
            counter_batches += 1

            # make training
            loss_rumors, cnt_correct = training_batch_iter(model, 'rumor', criterion, optimizer, device,
                                                           inputs_rumors, labels_rumors, h)
            cnt_correct_training_rumors += cnt_correct
            sum_loss_training_rumors += loss_rumors.item()

            loss_stances, cnt_correct = training_batch_iter(model, 'stance', criterion, optimizer, device,
                                                            inputs_stances, labels_stances, h)
            cnt_correct_training_stances += cnt_correct
            sum_loss_training_stances += loss_stances.item()

            # make validation and save the model if it is the best until now
            if i > 3:  # start validation only from epoch 5
                if 1 == counter_batches:
                    print('Validation of model: ')
                cnt_correct_r, cnt_correct_s = validation_or_testing(model, val_loader_rumors, val_loader_stances,
                                                                     criterion, device, i+1, validation_min_loss,
                                                                     loss_rumors, loss_stances, counter_batches, last_save)
                cnt_correct_validation_rumors += cnt_correct_r
                cnt_correct_validation_stances += cnt_correct_s

        # print accuracy and loss of the training
        training_loss_rumors = sum_loss_training_rumors / counter_batches
        print("Training loss rumors: {:.3f}".format(training_loss_rumors))
        training_acc_rumors = cnt_correct_training_rumors / (batch_size_training_rumors * counter_batches)
        print("Training accuracy rumors: {:.3f}%".format(training_acc_rumors * 100))

        training_loss_stances = sum_loss_training_stances / counter_batches
        print("Training loss stances: {:.3f}".format(training_loss_stances))
        training_acc_stances = cnt_correct_training_stances / (batch_size_training_stances * counter_batches)
        print("Training accuracy stances: {:.3f}%".format(training_acc_stances * 100))

        # print accuracy of the validation
        if i > 3:
            print('-----------------------------------------')

            validation_acc_rumors = cnt_correct_validation_rumors / (len(val_loader_rumors.dataset) * counter_batches)
            print("Validation accuracy rumors: {:.3f}%".format(validation_acc_rumors * 100))

            validation_acc_stances = cnt_correct_validation_stances / (len(val_loader_stances.dataset) * counter_batches)
            print("Validation accuracy stances: {:.3f}%".format(validation_acc_stances * 100))

            print('-----------------------------------------')

            print('Last save for model: epoch ' + str(last_save['last save']))

        # check time so far
        finish_time = gmtime()
        finish_time = strftime("%H:%M:%S", finish_time)
        formats = "%H:%M:%S"
        time_so_far = datetime.strptime(finish_time, formats) - datetime.strptime(start_time, formats)
        print('-----------------------------------------')
        print("Total runtime: ", time_so_far)
        print('-----------------------------------------')


def training_batch_iter(model, task_name, criterion, optimizer, device, inputs_batch, labels_batch, h):
    """
    Makes the forward step of specific task and returns the loss and number of correct predictions
    :param model:           the multi-task model
    :param task_name:       'rumor' or 'stances'
    :param criterion:       the loss function
    :param optimizer:       the optimizer
    :param device:          'cpu' or 'gpu'
    :param inputs_batch:    the inputs batch
    :param labels_batch:    the target labels batch
    :param h:               the initial 'h_t's
    :return:                - loss of the batch
                            - number of correct predictions
    """
    # set initial 'h' vectors of model's GRUs
    h_prev_task_rumors, h_prev_task_stances, h_prev_shared = h

    inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)

    # Clear gradients parameters
    optimizer.zero_grad()

    # Forward pass to get outputs of the model
    if 'rumor' == task_name:
        outputs, h_prev_shared, h_prev_task_rumors = model(inputs_batch, h_prev_shared, df.task_rumors_no,
                                                           h_prev_rumors=h_prev_task_rumors)
    else:  # 'stance' == task_name
        outputs, h_prev_shared, h_prev_task_stances = model(inputs_batch, h_prev_shared, df.task_stances_no,
                                                            h_prev_stances=h_prev_task_stances)

    # Calculate Loss
    if loss_function == 'BCELoss' or loss_function == 'MSELoss':
        loss = criterion(outputs, labels_batch.float())
    elif loss_function == 'L1Loss':
        loss = criterion(outputs, labels_batch)
    else:  # the default
        loss = criterion(outputs, (torch.max(labels_batch, 1)[1]).to(device))

    # Getting gradients parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    # count the number of correct outputs
    num_correct = count_correct(outputs, labels_batch, task_name, device)

    return loss, num_correct


def validation_or_testing(model, data_loader_rumors, data_loader_stances, criterion, device, epoch_no=None,
                          min_loss_dict=None, loss_train_r=None, loss_train_s=None, batch_no=None, last_save_dict=None,
                          operation='validation'):
    """
    Makes validation on specific task. Saves the dict of the model if it gave the best results so far.
    Returns the number of correct predictions
    :param model:                       the multi-task model
    :param data_loader_rumors:          DataLoader of rumor detection task
    :param data_loader_stances:         DataLoader of stance detection task
    :param criterion:                   the loss function
    :param device:                      'cpu' or 'gpu'
    :param epoch_no:                    epoch no
    :param min_loss_dict:               dictionary that contains the min losses of each task
    :param loss_train_r :               the loss of the training at this point of time for rumor detection task
    :param loss_train_s :               the loss of the training at this point of time for stance detection task
    :param batch_no:                    batch no
    :param last_save_dict               dictionary containing the last epoch where a save happened for each task
    :param operation                    validation' or 'testing'
    :return:                            number of correct predictions
    """
    all_losses_r = []  # for rumor detection task
    all_losses_s = []  # for stance detection task

    model.eval()  # set the model to evaluation mode

    sum_correct_r = 0  # for rumor detection task
    sum_correct_s = 0  # for stance detection task

    total_out_r = []  # for rumor detection task
    total_lab_r = []  # for rumor detection task
    total_out_s = []  # for stance detection task
    total_lab_s = []  # for stance detection task

    # get initial 'h' vectors of model's GRUs
    h_prev_task_rumors_val, h_prev_task_stances_val, h_prev_shared_val = model.init_hidden()

    # iterate through the batch
    for (inputs_rumors, labels_rumors), (inputs_stances, labels_stances) \
            in zip(data_loader_rumors, data_loader_stances):
        inputs_rumors, labels_rumors = inputs_rumors.to(device), labels_rumors.to(device)
        inputs_stances, labels_stances = inputs_stances.to(device), labels_stances.to(device)

        # Forward pass for rumor task, to get outputs of the model
        out_r, h_prev_shared_val, h_prev_task_rumors_val = model(inputs_rumors, h_prev_shared_val, df.task_rumors_no,
                                                                 h_prev_rumors=h_prev_task_rumors_val)
        # Forward pass for stance task, to get outputs of the model
        out_s, h_prev_shared_val, h_prev_task_stances_val = model(inputs_stances, h_prev_shared_val, df.task_stances_no,
                                                                  h_prev_stances=h_prev_task_stances_val)

        # we need this for calculation of F1 scores. we do it only for testing
        if 'testing' == operation:
            total_out_r += (torch.max(out_r, 1)[1]).to('cpu').tolist()
            total_lab_r += (torch.max(labels_rumors, 1)[1]).to('cpu').tolist()
            total_out_s += (torch.max(out_s, 1)[1]).to('cpu').tolist()
            total_lab_s += (torch.max(labels_stances, 1)[1]).to('cpu').tolist()

        # count the number of correct outputs
        sum_correct_r += count_correct(out_r, labels_rumors, 'rumor', device)
        sum_correct_s += count_correct(out_s, labels_stances, 'stance', device)

        # Calculate Loss
        if loss_function == 'BCELoss' or loss_function == 'MSELoss':
            loss_r = criterion(out_r, labels_rumors.float())
            loss_s = criterion(out_s, labels_stances.float())
        elif loss_function == 'L1Loss':
            loss_r = criterion(out_r, labels_rumors)
            loss_s = criterion(out_s, labels_stances)
        else:  # the default
            loss_r = criterion(out_r, (torch.max(labels_rumors, 1)[1]).to(device))
            loss_s = criterion(out_s, (torch.max(labels_stances, 1)[1]).to(device))
        all_losses_r.append(loss_r.item())
        all_losses_s.append(loss_s.item())

    # calculation of F1 scores
    if 'testing' == operation:
        # print F1 micro and macro scores for rumor detection
        score_f1_micro = f1_score(total_lab_r, total_out_r, average='micro')
        score_f1_macro = f1_score(total_lab_r, total_out_r, average='macro')
        print("For rumor detection:")
        print("F1 micro score: {:.3f}".format(score_f1_micro))
        print("F1 macro score: {:.3f}\n".format(score_f1_macro))

        # print F1 micro and macro scores for stance detection
        score_f1_micro = f1_score(total_lab_s, total_out_s, average='micro')
        score_f1_macro = f1_score(total_lab_s, total_out_s, average='macro')
        print("For stance detection:")
        print("F1 micro score: {:.3f}".format(score_f1_micro))
        print("F1 macro score: {:.3f}".format(score_f1_macro))

    if 'validation' == operation:
        print_and_save(model, epoch_no, batch_no, loss_train_r, loss_train_s, all_losses_r, all_losses_s, min_loss_dict,
                       last_save_dict)

    return sum_correct_r, sum_correct_s


def print_and_save(model, epoch_no, batch_no, loss_train_r, loss_train_s, all_losses_r, all_losses_s, min_loss_dict,
                   last_save_dict):
    """
    Prints the details of the validation and saves the dict of the model if it gave the best results so far.
    :param model:                       the multi-task model
    :param epoch_no:                    epoch no
    :param batch_no:                    batch no
    :param loss_train_r:                the loss of the training for rumor detection task
    :param loss_train_s:                the loss of the training for stance detection task
    :param all_losses_r:                list with all the losses of the validation for rumor detection task
    :param all_losses_s:                list with all the losses of the validation for stance detection task
    :param min_loss_dict:               dictionary that contains the min losses of each task
    :param last_save_dict               dictionary containing the the last epoch where a save happened for each task
    :return:                            void
    """
    model.train()  # set the model to train mode

    val_loss_avg = (np.mean(all_losses_r) + np.mean(all_losses_s)) / 2
    print("Epoch: {}/{}...".format(epoch_no, epochs),
          "batch: {}\n".format(batch_no),
          "Loss train for rumors: {:.6f}...".format(loss_train_r.item()),
          "Loss train for stances: {:.6f}\n".format(loss_train_s.item()),
          "Val Loss for rumors: {:.6f}".format(np.mean(all_losses_r)),
          "Val Loss for stances: {:.6f}\n".format(np.mean(all_losses_s)),
          "Val Loss avg: {:.6f}".format(val_loss_avg))
    if val_loss_avg <= min_loss_dict['min loss']:
        torch.save(model.state_dict(), 'model/model_state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(min_loss_dict['min loss'],
                                                                                          val_loss_avg))
        min_loss_dict['min loss'] = val_loss_avg
        last_save_dict['last save'] = epoch_no
    else:
        print()


def count_correct(outputs, labels_batch, task_name, device):
    """
    Counts the number of correct outputs (predictions)
    :param outputs:         the predictions of the model
    :param labels_batch:    the labels (targets)
    :param task_name:       'rumor' or 'stances'
    :param device:          'cpu' or 'gpu'
    :return:
    """
    num_correct = 0
    for out, label in zip(outputs, labels_batch):
        max_idx = torch.argmax(out, dim=0)  # dim=0 means: look in the first row
        if 'rumor' == task_name:
            one_hot = torch.nn.functional.one_hot(torch.tensor([max_idx]), df.output_dim_rumors)
        else:  # 'stance' == task_name
            one_hot = torch.nn.functional.one_hot(torch.tensor([max_idx]), df.output_dim_stances)
        one_hot = torch.squeeze(one_hot, 0)
        one_hot = one_hot.to(device)
        if torch.equal(label, one_hot):
            num_correct += 1

    return num_correct


if __name__ == '__main__':
    main()

