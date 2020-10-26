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

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle


fasttext.FastText.eprint = print

skipgram_path = os.path.join('embedding', 'model_skipgram.bin')


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
    loader_rumors = DataLoader(data_rumors, shuffle=False, batch_size=int(len(rumors_list)/2))

    # for stances
    data_replies = TensorDataset(torch.from_numpy(replies_list_embedded))
    loader_replies = DataLoader(data_replies, shuffle=False, batch_size=int(len(replies_list)/2))

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
    model_multi_task.load_state_dict(torch.load(os.path.join('model', 'model_state_dict.pt')))

    # Load hidden states (get initial 'h' vectors of model's GRUs)
    try:
        with open(os.path.join('model', 'h_prevs.pickle'), 'rb') as fp:
            h_training = pickle.load(fp)
            h_training = (torch.from_numpy(h_training['h_1']).to(device),
                          torch.from_numpy(h_training['h_2']).to(device),
                          torch.from_numpy(h_training['h_3']).to(device))
    except EnvironmentError:
        h_training = model_multi_task.init_hidden()
    h_prev_task_rumors_val, h_prev_task_stances_val, h_prev_shared_val = h_training

    out_r = []  # results for rumor detection task
    out_s = []  # results for stance detection task

    # run the model
    for inputs_rumors, inputs_replies in zip(loader_rumors, loader_replies):
        inputs_rumors, inputs_replies = inputs_rumors[0].to(device), inputs_replies[0].to(device)

        for i in range(1):
            # Forward pass for rumor task, to get outputs of the model
            output, h_prev_shared_val, h_prev_task_rumors_val = model_multi_task(inputs_rumors,
                                                                                h_prev_shared_val,
                                                                                df.task_rumors_no,
                                                                                h_prev_rumors=h_prev_task_rumors_val)
            out_r += output
            # Forward pass for stance task, to get outputs of the model
            output, h_prev_shared_val, h_prev_task_stances_val = model_multi_task(inputs_replies,
                                                                                 h_prev_shared_val,
                                                                                 df.task_stances_no,
                                                                                 h_prev_stances=h_prev_task_stances_val)
            out_s += output

    # print results for rumors
    print('\nPredictions for rumors task: ')
    true_rumors = []
    false_rumors = []
    unverified_rumors = []

    for output in out_r:
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

    for i, (tweet, output) in enumerate(zip(rumors, out_r)):
        print('Source tweet {}: {}'.format(i + 1, tweet))
        print('True rumor: {:.2f} | False rumor: {:.2f} | Unverified: {:.2f} \n'.format((output[0]*100),
                                                                                        (output[1]*100),
                                                                                        (output[2]*100)))

    # print results for stances
    print('\nPredictions for stances task: ')

    support_stances = []
    deny_stances = []
    query_stances = []
    commenting_stances = []

    for output in out_s:
        support_stances.append(output[0].item())
        deny_stances.append(output[1].item())
        query_stances.append(output[2].item())
        commenting_stances.append(output[3].item())

    avg_support_stances = np.mean(support_stances) * 100
    avg_deny_stances = np.mean(deny_stances) * 100
    avg_query_stances = np.mean(query_stances) * 100
    avg_commenting_stances = np.mean(commenting_stances) * 100

    # plot the results
    left = [1, 2, 3, 4]  # x-coordinates of left sides of bars
    height = [avg_support_stances, avg_deny_stances, avg_query_stances, avg_commenting_stances]  # heights of bars
    tick_label = ['Supporting', 'Denying', 'Questioning', 'Commenting']  # labels for bars

    # plotting a bar chart
    plt.bar(left, height, tick_label=tick_label, width=0.8, color=['green', 'red', 'purple', 'orange'])
    plt.ylabel('Stances distribution')
    plt.show()

    for i, (tweet, output) in enumerate(zip(replies, out_s)):
        print('Reply {}: {}'.format(i + 1, tweet))
        print('Supporting: {:.2f} | Deny: {:.2f} | Query: {:.2f} | Comment: {:.2f}\n'.format((output[0] * 100),
                                                                                             (output[1] * 100),
                                                                                             (output[2] * 100),
                                                                                             (output[3] * 100)))

    result_means = [round(avg_support_stances, 2), round(avg_deny_stances, 2),
                    round(avg_query_stances, 2), round(avg_commenting_stances, 2)]
    actual_means = [50, 10, 0, 40]

    x = np.arange(len(tick_label))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, result_means, width, label='Results')
    rects2 = ax.bar(x + width / 2, actual_means, width, label='Actual')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Stances distribution')
    ax.set_title('Results accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_label)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.tight_layout()
    plt.show()

def main():
    # load the fasttext model
    if not os.path.isfile(skipgram_path):
        print('Skipgram model not found')
        sys.exit()
    else:
        skipgram_model = fasttext.load_model(skipgram_path)

    rumors = ['#breaking Senior IOC member Dick Pound tells @usatodaysports the Tokyo 2020 Olympic Games will be postponed amid the coronavirus pandemic: "It will come in stages. We will postpone this and begin to deal with all the ramifications of moving this, which are immense." @jaketapper',
              'The International Olympic Committee has announced the upcoming 2020 Tokyo Olympic Games will be postponed, via USA Today.',
              'I’m hearing the Olympic Committee will likely postpone the games for a year or two but not cancel. Maybe we could go back to doing winter and summer in the same year I liked that better.',
              'Senior figures in British sport are now “90 per cent certain” that the Olympic Games in Tokyo will be postponed.',
              'And there it is, the Olympic Games as postponed...',
              'Japanese Officials Say 2020 Olympic Games Could be Postponed.',
              'Coronavirus: Olympic Games to be postponed officially any day now as 2020 organisers face up to the inevitable.',
              'Coronavirus will not postpone the Olympics, IOC says.']
    replies = ['It’s the right thing. I feel badly but it’s not worth the risks',
               'Should the Tokyo Olympic be cancelled rather than postponed, no country would try to host such a huge international event with enormous risk. If the Olympic is a noble event with worldwide commitments, would not it be more constructive to discuss "how to protect the Olympics"?',
               'No. "Senior IOC member Dick Pound." This is not real and I will not fall for this.',
               'To make it happen, Japan needs to aggressively start testing, tracing, isolating and treating people as well as informing people in utmost sincere and transparent way. Otherwise, this outbreak will never end as the world is interconnected and interdependent.',
               'Finally!!',
               'Smart move. If they did not postpone, they would have an Olympics with high school JV team swimmers and empty stands.',
               'Good decision.',
               'Ok now I know it’s serious.',
               'Good news. I watch olympics but we need to shut everything down at least until new cases drops dramatically across the globe.',
               'When reached for comment regarding the difficult of the decision, Dick Pound stated simply, “It was hard.”']

    examine_example(skipgram_model,
                    rumors,
                    replies
                    )


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



if __name__ == '__main__':
    main()

