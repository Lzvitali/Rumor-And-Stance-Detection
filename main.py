"""
In this file we will examine our model on that tweet: https://mobile.twitter.com/willripleyCNN/status/1242202078786785281
and plot the results
"""
from model.models import GRUMultiTask
import data.data as dt
import defines as df
import fasttext
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

fasttext.FastText.eprint = print

skipgram_path = 'embedding\\model_skipgram.bin'


def examine_example(fasttext_model, rumors, replies):
    rumors_list = []
    for rumor in rumors:
        rumor = dt.tweet_cleaner(rumor)
        rumors_list.append(rumor)

    replies_list = []
    for reply in replies:
        reply = dt.tweet_cleaner(reply)
        replies_list.append(reply)

    rumors_list_embedded = np.zeros((len(rumors_list), df.input_length), dtype=np.float32)
    replies_list_embedded = np.zeros((len(replies_list), df.input_length), dtype=np.float32)

    for i, rumor in enumerate(rumors_list):
        rumors_list_embedded[i, :] = fasttext_model.get_sentence_vector(rumor)

    for i, reply in enumerate(replies_list):
        replies_list_embedded[i, :] = fasttext_model.get_sentence_vector(reply)

    # load our multi-task model
    # for rumors
    data_rumors = TensorDataset(torch.from_numpy(rumors_list_embedded))
    loader_rumors = DataLoader(data_rumors, shuffle=False, batch_size=len(rumors_list))

    # for stances
    data_replies = TensorDataset(torch.from_numpy(replies_list_embedded))
    loader_replies = DataLoader(data_replies, shuffle=False, batch_size=len(replies_list))

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

    # Loading the model
    model_multi_task.load_state_dict(torch.load('model/model_state_dict.pt'))

    # get initial 'h' vectors of model's GRUs
    h_prev_task_rumors_val, h_prev_task_stances_val, h_prev_shared_val = model_multi_task.init_hidden()

    out_r = []  # results for rumor detection task
    out_s = []  # results for stance detection task

    # run the model
    for inputs_rumors, inputs_replies in zip(loader_rumors, loader_replies):
        inputs_rumors, inputs_replies = inputs_rumors[0].to(device), inputs_replies[0].to(device)

        for i in range(1):
            # Forward pass for rumor task, to get outputs of the model
            out_r, h_prev_shared_val, h_prev_task_rumors_val = model_multi_task(inputs_rumors,
                                                                                h_prev_shared_val,
                                                                                df.task_rumors_no,
                                                                                h_prev_rumors=h_prev_task_rumors_val)
            # Forward pass for stance task, to get outputs of the model
            out_s, h_prev_shared_val, h_prev_task_stances_val = model_multi_task(inputs_replies,
                                                                                 h_prev_shared_val,
                                                                                 df.task_stances_no,
                                                                                 h_prev_stances=h_prev_task_stances_val)

    # print results for rumors
    print('\nPredictions for rumors task: ')
    true_rumors = []
    false_rumors = []
    unverified_rumors = []
    for i, (tweet, output) in enumerate(zip(rumors, out_r)):
        print('Source tweet{}: {}'.format(i, tweet))
        print('True rumor: {:.2f} | False rumor: {:.2f} | Unverified: {:.2f} \n'.format((output[0]*100),
                                                                                        (output[1]*100),
                                                                                        (output[2]*100)))
        true_rumors.append(output[0].item())
        false_rumors.append(output[1].item())
        unverified_rumors.append(output[2].item())

    avg_true_rumor = np.mean(true_rumors) * 100
    avg_false_rumor = np.mean(false_rumors) * 100
    avg_true_unverified = np.mean(unverified_rumors) * 100

    # plot the results
    slices = [avg_true_rumor, avg_false_rumor, avg_true_unverified]
    activities = ['True rumor', 'False rumor', 'Unverified rumor']
    plt.pie(slices, labels=activities, startangle=180, shadow=True, explode=(0.1, 0, 0), autopct='%1.1f%%')
    plt.legend(loc="center left", bbox_to_anchor=(0.85, 0.1, 0.5, 0.85))
    plt.show()

    # print results for stances
    print('\nPredictions for stances task: ')
    counters = [0, 0, 0, 0]
    for i, (tweet, output) in enumerate(zip(replies, out_s)):
        print('Reply{}: {}'.format(i, tweet))
        print('Supporting: {:.2f} | Deny: {:.2f} | Query: {:.2f} | Comment: {:.2f}\n'.format((output[0] * 100),
                                                                                             (output[1] * 100),
                                                                                             (output[2] * 100),
                                                                                             (output[3] * 100)))
        counters[torch.argmax(output, dim=0).item()] += 1

    # plot the results
    left = [1, 2, 3, 4]  # x-coordinates of left sides of bars
    height = [counters[0], counters[1], counters[2], counters[3]]  # heights of bars
    tick_label = ['Supporting', 'Denying', 'Questioning', 'Commenting']  # labels for bars

    # plotting a bar chart
    plt.bar(left, height, tick_label=tick_label, width=0.8, color=['green', 'red', 'purple', 'orange'])
    plt.show()


def main():
    # load the fasttext model
    if not os.path.isfile(skipgram_path):
        print('Skipgram model not found')
        sys.exit()
    else:
        skipgram_model = fasttext.load_model(skipgram_path)

    rumors = ['#breaking Senior IOC member Dick Pound tells @usatodaysports the Tokyo 2020 Olympic Games will be postponed amid the coronavirus pandemic: "It will come in stages. We will postpone this and begin to deal with all the ramifications of moving this, which are immense." @jaketapper']
    replies = ['It’s the right thing. I feel badly but it’s not worth the risks',
               'I truly believe this Olympic should be protected by international efforts. Future-oriented options.',
               'Who now?',
               'No. "Senior IOC member Dick Pound." This is not real and I will not fall for this.']

    examine_example(skipgram_model,
                    rumors,
                    replies
                    )


if __name__ == '__main__':
    main()

