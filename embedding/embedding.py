"""
This script converts the csv data that 'data.py' generates into '.npy' files (after embedding with FastText)
Then it expands each '.npy' file (of rumor detection task) with data from 'Twitter15-16' dataset.
Twitter15-16' dataset can be found here:
"""

import fasttext
import os
import csv
import numpy as np
import defines as df
import data.data as dt

# Preprocessed data folders paths
preprocessed_data_paths_RumourEval = {
    'training path':    '..\\data\\preprocessed data\\training',
    'validation path':  '..\\data\\preprocessed data\\validation',
    'test path':        '..\\data\\preprocessed data\\test'
}

data_paths_twitter15_16 = {
    'twitter15-16 tweets': '..\\data\\raw data\\twitter15-16\\source_tweets.txt',
    'twitter15-16 labels': '..\\data\\raw data\\twitter15-16\\labels.txt',
}

# 'text9' is cleaned the data of the English 'Wikipedia 9' dataset: enwik9. The data is UTF-8 encoded XML consisting
# primarily of English text. enwik9 contains 243,426 article titles, of which 85,560 are #REDIRECT to fix broken links,
# and the rest are regular articles.
wiki_dataset = '..\\..\\fasttextWiki\\text9'

skipgram_path = 'model_skipgram.bin'


# labels to np.array
# for rumors
label_true = np.array([1, 0, 0], dtype=np.int64)
label_false = np.array([0, 1, 0], dtype=np.int64)
label_unverified = np.array([0, 0, 1], dtype=np.int64)
# for stances
label_support = np.array([1, 0, 0, 0], dtype=np.int64)
label_deny = np.array([0, 1, 0, 0], dtype=np.int64)
label_query = np.array([0, 0, 1, 0], dtype=np.int64)
label_comment = np.array([0, 0, 0, 1], dtype=np.int64)


def prepare_data_from_txt(fasttext_model):
    counters_training = {
        'total rumors': 0,
        'true rumor': 0,
        'false rumor': 0,
        'unverified rumor': 0,
    }
    counters_validation = {
        'total rumors': 0,
        'true rumor': 0,
        'false rumor': 0,
        'unverified rumor': 0,
    }
    counters_test = {
        'total rumors': 0,
        'true rumor': 0,
        'false rumor': 0,
        'unverified rumor': 0,
    }

    # Opening the files
    try:
        tweets_file = open(data_paths_twitter15_16['twitter15-16 tweets'], mode='r', encoding="utf8")
        labels_file = open(data_paths_twitter15_16['twitter15-16 labels'], mode='r', encoding="utf8")
    except IOError:
        print('error')
        return

    tweets_file_data = tweets_file.readlines()
    labels_file_data = labels_file.readlines()

    # check the number of relevant tweets
    cnt_relevant_tweets = 0
    for tweet_line, label_line in zip(tweets_file_data, labels_file_data):
        label_end = label_line.find(':')
        label = label_line[:label_end]
        # if the label is 'non-rumor' it is not relevant for us - so don't count it
        if label != 'non-rumor':
            cnt_relevant_tweets += 1
        tweets_validation = np.zeros((172, df.input_length), dtype=np.float32)
        labels_validation = np.zeros((172, df.output_dim_rumors), dtype=np.int64)

        tweets_test = np.zeros((144, df.input_length), dtype=np.float32)
        labels_test = np.zeros((144, df.output_dim_rumors), dtype=np.int64)

        tweets_training = np.zeros((cnt_relevant_tweets - 172 - 144, df.input_length), dtype=np.float32)
        labels_training = np.zeros((cnt_relevant_tweets - 172 - 144, df.output_dim_rumors), dtype=np.int64)

        i = 0
        for tweet_line, label_line in zip(tweets_file_data, labels_file_data):
            label_end = label_line.find(':')
            label = label_line[:label_end]

        # if the label is 'non-rumor' it is not relevant for us - so go to the next line
        if label == 'non-rumor':
            continue

        tweet_start = tweet_line.find('\t')
        tweet = tweet_line[tweet_start:]

        # clean the tweet text
        tweet = dt.tweet_cleaner(tweet)

        label = label.lower()

        if label == 'true':
            label_v = label_true
            label_name = 'true rumor'
        elif label == 'false':
            label_v = label_false
            label_name = 'false rumor'
        else:  # label == 'unverified':
            label_v = label_unverified
            label_name = 'unverified rumor'

        # add to the validation
        if i < 172:
            tweets_validation[i, :] = fasttext_model.get_sentence_vector(tweet)
            labels_validation[i, :] = label_v
            counters_validation['total rumors'] += 1
            counters_validation[label_name] += 1

        # add to the test
        elif 172 <= i < (172 + 144):
            tweets_test[i - 172, :] = fasttext_model.get_sentence_vector(tweet)
            labels_test[i - 172, :] = label_v
            counters_test['total rumors'] += 1
            counters_test[label_name] += 1
        # add to the training
        else:
            tweets_training[i - 172 - 144, :] = fasttext_model.get_sentence_vector(tweet)
            labels_training[i - 172 - 144, :] = label_v
            counters_training['total rumors'] += 1
            counters_training[label_name] += 1

        i += 1

    # print the amount of added tweets
    print('\nFrom Twitter15-16 dataset:')
    print('For training: {}'.format(counters_training))
    print('For validation: {}'.format(counters_validation))
    print('For test: {}'.format(counters_test))

    # load the previous data, concatenate with the new and save it
    # validation - tweets
    previous = np.load(os.path.join(preprocessed_data_paths_RumourEval['validation path'], 'rumors_tweets.npy')) \
        if os.path.isfile(os.path.join(preprocessed_data_paths_RumourEval['validation path'], 'rumors_tweets.npy')) \
        else []  # get data if exist
    np.save(os.path.join(preprocessed_data_paths_RumourEval['validation path'], 'rumors_tweets.npy'),
            np.concatenate([previous, tweets_validation]))  # save the new

    # validation - labels
    previous = np.load(os.path.join(preprocessed_data_paths_RumourEval['validation path'], 'rumors_labels.npy')) \
        if os.path.isfile(os.path.join(preprocessed_data_paths_RumourEval['validation path'], 'rumors_labels.npy')) \
        else []  # get data if exist
    np.save(os.path.join(preprocessed_data_paths_RumourEval['validation path'], 'rumors_labels.npy'),
            np.concatenate([previous, labels_validation]))  # save the new

    print('\n172 rumor tweets added to: Validation set')

    # test - tweets
    previous = np.load(os.path.join(preprocessed_data_paths_RumourEval['test path'], 'rumors_tweets.npy')) \
        if os.path.isfile(os.path.join(preprocessed_data_paths_RumourEval['test path'], 'rumors_tweets.npy')) \
        else []  # get data if exist
    np.save(os.path.join(preprocessed_data_paths_RumourEval['test path'], 'rumors_tweets.npy'),
            np.concatenate([previous, tweets_test]))  # save the new

    # test - labels
    previous = np.load(os.path.join(preprocessed_data_paths_RumourEval['test path'], 'rumors_labels.npy')) \
        if os.path.isfile(os.path.join(preprocessed_data_paths_RumourEval['test path'], 'rumors_labels.npy')) \
        else []  # get data if exist
    np.save(os.path.join(preprocessed_data_paths_RumourEval['test path'], 'rumors_labels.npy'),
            np.concatenate([previous, labels_test]))  # save the new

    print('144 rumor tweets added to: Test set')

    # training - tweets
    previous = np.load(os.path.join(preprocessed_data_paths_RumourEval['training path'], 'rumors_tweets.npy')) \
        if os.path.isfile(os.path.join(preprocessed_data_paths_RumourEval['training path'], 'rumors_tweets.npy')) \
        else []  # get data if exist
    np.save(os.path.join(preprocessed_data_paths_RumourEval['training path'], 'rumors_tweets.npy'),
            np.concatenate([previous, tweets_training]))  # save the new

    # training - labels
    previous = np.load(os.path.join(preprocessed_data_paths_RumourEval['training path'], 'rumors_labels.npy')) \
        if os.path.isfile(os.path.join(preprocessed_data_paths_RumourEval['training path'], 'rumors_labels.npy')) \
        else []  # get data if exist
    np.save(os.path.join(preprocessed_data_paths_RumourEval['training path'], 'rumors_labels.npy'),
            np.concatenate([previous, labels_training]))  # save the new

    print('{} rumor tweets added to: Training set'.format((cnt_relevant_tweets - 172 - 144)))

    tweets_file.close()
    labels_file.close()


def prepare_data_from_csv(base_path, csv_path, output_dim, fasttext_model, counters):
    """
    Gets CSV with tweets and labels and creates np.arrays with embedded tweets and labels
    :param base_path:       path to the folder
    :param csv_path:        path to the csv file
    :param output_dim:      will be 3 for rumor detection and 4 for stance detection
    :param fasttext_model:  the skip-gram model generated by FastText
    :param counters:        dictionary holding 2 counters, one for each task
    :return: void
    """
    with open(csv_path, 'r') as csv_file:
        num_of_tweets = sum(1 for line in csv_file)
        num_of_tweets -= 1

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # skip the heading

        if output_dim == df.output_dim_rumors:
            set_name = 'rumors'
        else:
            set_name = 'stances'

        tweets = np.zeros((num_of_tweets, df.input_length), dtype=np.float32)
        labels = np.zeros((num_of_tweets, output_dim), dtype=np.int64)

        for i, row in enumerate(csv_reader):
            tweets[i, :] = fasttext_model.get_sentence_vector(row[0])
            label = row[1].lower()

            if set_name == 'rumors':
                counters['total rumors'] += 1
                if label == 'true':
                    labels[i, :] = label_true
                    counters['true rumor'] += 1
                elif label == 'false':
                    labels[i, :] = label_false
                    counters['false rumor'] += 1
                elif label == 'unverified':
                    labels[i, :] = label_unverified
                    counters['unverified rumor'] += 1
            else:
                counters['total stances'] += 1
                if label == 'support':
                    labels[i, :] = label_support
                    counters['support'] += 1
                elif label == 'deny':
                    labels[i, :] = label_deny
                    counters['deny'] += 1
                elif label == 'query':
                    labels[i, :] = label_query
                    counters['query'] += 1
                elif label == 'comment':
                    labels[i, :] = label_comment
                    counters['comment'] += 1

        np.save(os.path.join(base_path, set_name + '_tweets.npy'), tweets)
        np.save(os.path.join(base_path, set_name + '_labels.npy'), labels)


def main():
    if not os.path.isfile(skipgram_path):
        model = fasttext.train_unsupervised(wiki_dataset,
                                            "skipgram",
                                            minn=3,  # min length of char ngram - subword between 3 and 6 characters is as popular
                                            maxn=6,  # max length of char ngram
                                            dim=df.input_length,  # size of word vectors - any value in the 100-300 range is as popular
                                            lr=0.05  # learning rate - The default value is 0.05 which is a good compromise. If you want to play with it we suggest to stay in the range of [0.01, 1]
                                            )
        model.save_model(skipgram_path)
    else:
        model = fasttext.load_model(skipgram_path)

    # ------------------------------------- From: RumourEval 2019 Dataset ---------------------------------------------

    # go through the dataset and use fasttest (for embedding) to create numpy arrays
    counters = {
        'total rumors': 0,
        'true rumor': 0,
        'false rumor': 0,
        'unverified rumor': 0,

        'total stances': 0,
        'support': 0,
        'deny': 0,
        'query': 0,
        'comment': 0,
    }

    # go through training data folder, validation data folder and test data folder
    for _, preprocessed_data_path in preprocessed_data_paths_RumourEval.items():
        counters['total rumors'] = 0
        counters['true rumor'] = 0
        counters['false rumor'] = 0
        counters['unverified rumor'] = 0
        counters['total stances'] = 0
        counters['support'] = 0
        counters['deny'] = 0
        counters['query'] = 0
        counters['comment'] = 0

        rumors_csv_path = os.path.join(preprocessed_data_path, 'rumors.csv')
        stances_csv_path = os.path.join(preprocessed_data_path, 'stances.csv')

        prepare_data_from_csv(preprocessed_data_path, rumors_csv_path, df.output_dim_rumors, model, counters)
        prepare_data_from_csv(preprocessed_data_path, stances_csv_path, df.output_dim_stances, model, counters)

        # print counters
        if 'training' in preprocessed_data_path:
            set_name = 'training'
        elif 'validation' in preprocessed_data_path:
            set_name = 'validation'
        else:
            set_name = 'test'
        print('{}: {}'.format(set_name, counters))

    # End: From: RumourEval 2019 Dataset ------------------------------------------------------------------------------

    # ------------------------------------- From: Tweeter 15 & 16 Dataset ---------------------------------------------

    prepare_data_from_txt(model)

    # End: From: Tweeter 15 & 16 Dataset ------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
    print('done')
