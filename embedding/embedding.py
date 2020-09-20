'''
TODO: add description
'''

import fasttext
import os
import numpy as np


# 'text9' is cleaned the data of the English 'Wikipedia 9' dataset: enwik9. The data is UTF-8 encoded XML consisting
# primarily of English text. enwik9 contains 243,426 article titles, of which 85,560 are #REDIRECT to fix broken links,
# and the rest are regular articles.
wiki_dataset = '..\\..\\fasttextWiki\\text9'

skipgram_path = 'model_skipgram.bin'


def main():
    if not os.path.isfile(skipgram_path):
        model = fasttext.train_unsupervised(wiki_dataset,
                                            "skipgram",
                                            minn=3,  # min length of char ngram - subword between 3 and 6 characters is as popular
                                            maxn=6,  # max length of char ngram
                                            dim=250,  # size of word vectors - any value in the 100-300 range is as popular
                                            lr=0.1  # learning rate - The default value is 0.05 which is a good compromise. If you want to play with it we suggest to stay in the range of [0.01, 1]
                                            )
        model.save_model(skipgram_path)
    else:
        model = fasttext.load_model(skipgram_path)

    # test
    d = model.get_sentence_vector("Please work")
    print(d)

    print(type(model))

    w = model.get_word_vector("King")
    print(w)


if __name__ == '__main__':
    main()