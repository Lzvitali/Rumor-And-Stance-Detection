"""
This Script converts this dataset: https://figshare.com/articles/RumourEval_2019_data/8845580
which described here: https://www.aclweb.org/anthology/S19-2147.pdf to CSV files, in the following logic:
It creates 3 folders: training, validation and testing.
Each folder contains 2 CSV files: rumors.csv and stances.csv.
each CSV file has 2 columns: tweet content and label.
"""

import os
import sys
import csv
import json
import pathlib
from cleantext import clean  # reference: https://github.com/jfilter/clean-text


# Raw data folders paths
raw_data_paths = {
    'training path':    'raw data\\rumoureval-2019-training-data',
    'validation path':  'raw data\\rumoureval-2019-training-data',
    'test path':        'raw data\\rumoureval-2019-test-data'
}

# jason with labels
raw_data_labels_paths = {
    'training path':    raw_data_paths['training path'] + '\\train-key.json',
    'validation path':  raw_data_paths['validation path'] + '\\dev-key.json',
    'test path':        raw_data_paths['test path'] + '\\final-eval-key.json'
}

# Preprocessed data folders paths
preprocessed_data_paths = {
    'training path':    'preprocessed data\\training',
    'validation path':  'preprocessed data\\validation',
    'test path':        'preprocessed data\\test'
}


def set_tweet_label(tweet_path, writer_rumors, writer_stances, tweet_id, rumors_labels, stances_labels, counters_dict):
    """
    Writes to the CSV files the "clean" tweet content and it's label
    :param tweet_path:      tweet full path
    :param writer_rumors:   object used for writing into the rumors csv
    :param writer_stances:  object used for writing into the stances csv
    :param tweet_id: tweet  id (as string)
    :param rumors_labels:   dictionary containing the labels for each tweet id that related to rumor detection task
    :param stances_labels:  dictionary containing the labels for each tweet id that related to stance classification task
    :return: void
    """
    # Opening JSON file
    with open(tweet_path, 'r') as tweet_file:
        # get JSON object as a dictionary
        tweet_dict = json.load(tweet_file)
        if 'text' in tweet_dict:
            tweet_content = tweet_dict['text']
            tweet_content = clean(tweet_content,
                                  fix_unicode=True,  # fix various unicode errors
                                  to_ascii=True,  # transliterate to closest ASCII representation
                                  lower=True,  # lowercase text
                                  no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                                  no_urls=True,  # replace all URLs with a special token
                                  no_emails=True,  # replace all email addresses with a special token
                                  no_phone_numbers=True,  # replace all phone numbers with a special token
                                  no_numbers=False,  # replace all numbers with a special token
                                  no_digits=False,  # replace all digits with a special token
                                  no_currency_symbols=True,  # replace all currency symbols with a special token
                                  no_punct=True,  # fully remove punctuation
                                  replace_with_url=" ",
                                  replace_with_email=" ",
                                  replace_with_phone_number=" ",
                                  replace_with_number=" ",
                                  replace_with_digit=" ",
                                  replace_with_currency_symbol=" ",
                                  lang="en")
            if tweet_id in rumors_labels:
                tweet_label = rumors_labels[tweet_id]
                row = [tweet_content, tweet_label]
                writer_rumors.writerow(row)
                counters_dict['rumors'] += 1

            if tweet_id in stances_labels:
                tweet_label = stances_labels[tweet_id]
                row = [tweet_content, tweet_label]
                writer_stances.writerow(row)
                counters_dict['stances'] += 1


def division_by_tasks():
    counters = {
        'rumors': 0,
        'stances': 0,
    }

    # go through training data folder, validation data folder and test data folder
    for (_, raw_data_path), (_, raw_data_labels_path), (_, preprocessed_data_path) \
            in zip(raw_data_paths.items(), raw_data_labels_paths.items(), preprocessed_data_paths.items()):
        counters['rumors'] = 0
        counters['stances'] = 0

        # Opening JSON file
        try:
            labels_file = open(raw_data_labels_path)
        except IOError:
            print("error")
            break

        # get JSON object as a dictionary
        labels = json.load(labels_file)

        try:
            rumors_labels = labels['subtaskbenglish']
        except KeyError:
            rumors_labels = {}

        try:
            stances_labels = labels['subtaskaenglish']
        except KeyError:
            stances_labels = {}

        # create folder for training / validation / test
        pathlib.Path(preprocessed_data_path).mkdir(parents=True, exist_ok=True)

        # create 2 CSV file: 1 for rumors , 1 for stances
        csv_fieldnames = ['Tweet content', 'Label']  # csv column names and content arrays
        # open CSV files
        try:
            csv_file_rumors = open(os.path.join(preprocessed_data_path, 'rumors.csv'), 'w', newline='')
            csv_writer_rumors = csv.writer(csv_file_rumors)
            csv_writer_rumors.writerow(csv_fieldnames)
            csv_file_stances = open(os.path.join(preprocessed_data_path, 'stances.csv'), 'w', newline='')
            csv_writer_stances = csv.writer(csv_file_stances)
            csv_writer_stances.writerow(csv_fieldnames)
        except:
            print('Exception with opening CSV files')
            sys.exit()

        dirs = os.listdir(raw_data_path)
        for dir_name in dirs:
            dir_fullPath = os.path.join(raw_data_path, dir_name)
            if os.path.isdir(dir_fullPath) and dir_name.startswith("twitter"):
                claim_dirs_path = dir_fullPath
                claim_dirs = os.listdir(claim_dirs_path)
                for claim_dir in claim_dirs:
                    claim_dir_fullPath = os.path.join(dir_fullPath, claim_dir)
                    if os.path.isdir(claim_dir_fullPath):
                        # scan the data folders and find all the tweets
                        in_claim_dirs_path = os.path.join(claim_dirs_path, claim_dir)
                        in_claim_dirs = os.listdir(in_claim_dirs_path)
                        for in_claim_dir in in_claim_dirs:
                            root_tweet_dirs_path = os.path.join(in_claim_dirs_path, in_claim_dir)
                            if os.path.isdir(root_tweet_dirs_path):
                                root_tweet_dirs = os.listdir(root_tweet_dirs_path)
                                for root_tweet_dir in root_tweet_dirs:
                                    root_tweet_dir_fullPath = os.path.join(root_tweet_dirs_path)
                                    if os.path.isdir(root_tweet_dir_fullPath) and 'source-tweet' == root_tweet_dir:
                                        tweet_full_path = os.path.join(root_tweet_dir_fullPath, root_tweet_dir,
                                                                       in_claim_dir + '.json')
                                        set_tweet_label(tweet_full_path, csv_writer_rumors, csv_writer_stances,
                                                        in_claim_dir, rumors_labels, stances_labels, counters)
                                    elif os.path.isdir(root_tweet_dir_fullPath) and 'replies' == root_tweet_dir:
                                        reply_tweet_files_path = os.path.join(root_tweet_dir_fullPath, root_tweet_dir)
                                        reply_tweet_files = os.listdir(reply_tweet_files_path)
                                        for reply_tweet_file in reply_tweet_files:
                                            reply_tweet_file_fullPath = os.path.join(reply_tweet_files_path,
                                                                                     reply_tweet_file)
                                            if not os.path.isdir(reply_tweet_file_fullPath):
                                                set_tweet_label(reply_tweet_file_fullPath, csv_writer_rumors,
                                                                csv_writer_stances, reply_tweet_file[:-5],
                                                                rumors_labels, stances_labels, counters)
        csv_file_rumors.close()
        csv_file_stances.close()
        labels_file.close()

        if 'training' in preprocessed_data_path:
            set_name = 'training'
        elif 'validation' in preprocessed_data_path:
            set_name = 'validation'
        else:
            set_name = 'test'
        print(set_name + ': ' + str(counters))


if __name__ == '__main__':
    division_by_tasks()
    print('Finished')