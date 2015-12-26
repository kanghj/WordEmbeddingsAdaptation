import glob
import os
import logging
import pickle
import multiprocessing

import nltk

from collections import Counter
from itertools import izip

"""
Counts word frequency,
and writes to file "token_idf"
"""

_config = {
        'directories_for_testing' : ["C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2000",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2001",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2002",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2003",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2004",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2005",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2006",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2007",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2008",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2009",
                                     "C:/Users/user/fyp/fyp_models/corpora/un/text/en-zh/2010",],
        'wordlst_path' : "C:/Users/user/fyp/ims_0.9.2/senna/hash/words.lst", # only words in this file (senna's vocabulary) matter
        }


def _load_wordlist():
    wordlst_path = _config['wordlst_path']
    with open(wordlst_path, 'r') as wordlst_file:
        for word in wordlst_file:
            WORDLIST.append(word.strip())

WORDLIST = []
_load_wordlist()

def process_intermediate_results(result):
    """
    Intermedaite processing, for senna, should convert all 4-digits numbers to ####, any numerical value to #,
    
    """

def get_words_and_frequency(document):
    with open(document) as opened_document:
        all_text = opened_document.read().lower()
        tokenised_text = all_text.split()

    return Counter(tokenised_text)


def document_freq_for_terms_in(directory):
    try:
        english_documents = glob.glob(directory + '/*en.snt')

        result = Counter()
        for document in english_documents:
            result = result + get_words_and_frequency(document)

        print 'completed', directory

        return result
    except Exception as e:
        logging.exception(e)
        print multiprocessing.current_process()
        raise e


if __name__ == '__main__':

    # first check if we already have the processed data:
    try:
        with open('idf_result.p', 'r') as pickle_file:
            print "getting pickled result"
            result = pickle.load(pickle_file)
        print "got pickled results"
    except:
        pool = multiprocessing.Pool(processes=6)
        uncompiled_results = pool.map(document_freq_for_terms_in, _config['directories_for_testing'])
        
        result = Counter()
        for partial_result in uncompiled_results:
            result += partial_result


        print "in main, "

        print "pickling combined results"
        pickle.dump(result, open( "idf_result.p", "wb" ))
    
    print "results length", len(result)

    print "write as a wordlst ordered file"
    with open( 'token_idf', 'w+') as idf_file:
        for word in WORDLIST:
            print word
            idf_file.write(str(result[word]))
            idf_file.write('\n')

    
    print 'All completed for '
    print _config