import csv
import xml.etree.cElementTree as ET
import codecs
import glob
import re
import logging

from multiprocessing import Pool
from itertools import izip

config = {
        'directories_for_testing' : ['C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/noun', 
                                      'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/verb',
                                      'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/adv',
                                      'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/adj'],
        'wordlst' : 'C:/Users/user/fyp/ims_0.9.2/senna/hash/words.lst'
        }

ENCODING = "iso-8859-1"


# maps words to index, this is based on the line number in words.lst
WORD_TO_INDEX = {}

#SENSE_TO_INDEX = {}


def construct_word_to_index(wordlst_file):
    for i, word in enumerate(wordlst_file):
        WORD_TO_INDEX[word.strip()] = i + 1 # + 1 to be 1-indexed, this is convenient for later use 
                                            # because the lua torch library is 1-indexed

with open(config['wordlst']) as wordlst_file:
    construct_word_to_index(wordlst_file)

class Instance(object):
    number = -1
    head = ""
    tail = ""
    label = ""
    label_as_int = 0

    training_instance_strategy = "sentence"

    @classmethod
    def get_context(cls, head, tail):
        if cls.training_instance_strategy == "raw":
            front = head
            back = tail
        elif cls.training_instance_strategy == "sentence":
            # get sentence
            # last sentence of head and
            front = head.split('.')[-1]
            # first sentence of tail
            back = tail.split('.')[0]
        elif cls.training_instance_strategy == "window":
            front = head.split(' ')[-3 : -1]
            back = tail.split(' ')[0 : 3]
        else:
            raise ValueError("invalid training strategy : " + \
                              cls.training_instance_strategy)
        return (front + ' ' + back).encode(ENCODING)

    def __init__(self, number, head, tail, label):
        self.number = number
        if head is None:
            head = ""
        if tail is None:
            tail = ""
        self.head = ' '.join(head.replace('\n','').split()).strip()
        self.tail = ' '.join(tail.replace('\n','').split()).strip()
        self.label = label 


    def __repr__(self):
        return "Instance[id:" + str(self.number) + \
               ", context: " + self.head + " " + self.tail + ", answer:" + \
                self.label + "]"
    
    def get_context_list(self):
        context_words = Instance.get_context(self.head, self.tail)
        context_indices = []
        for i, word in enumerate(context_words.split()):
            try:
                context_indices.append(WORD_TO_INDEX[word.lower()])
            except KeyError:
                context_indices.append(WORD_TO_INDEX['UNKNOWN'])
        return context_indices


def get_instances(xml_file, key_file):
    tree = ET.parse(xml_file)
    xml_instances = tree.getroot().findall('.//instance')

    ids     = map(lambda x: x.attrib['id'], xml_instances)
    heads   = map(lambda x: x.find('context').text, xml_instances)
    # the part of the context behind the <head> doesn't get included in head
    # so we use the tail of the head to obtain it
    tails   = map(lambda x: x.find('.//head').tail, xml_instances)

    with open(key_file) as labels_file:
        labels = [line.split(' ')[2] for line in labels_file]
        # labels is now the sense in wordnet, in the senseval format
        # but we should convert it into a numbered format, based on a key file
        

    csv_instances = []
    for number, context, tail, label in izip(ids, heads, tails, labels):
        instance = Instance(number, context, tail, label)
        #print instance
        csv_instances.append(instance)

    return csv_instances

def write_csv_for_directory(directory):
    try:
        with open('./' + directory.split('/')[-1] + '.csv', 'wb+' )  as output_file:
            writer = csv.writer(output_file)

            # inputs
            xml_files = glob.glob(directory + '/*.xml')
            key_files = glob.glob(directory + '/*.key')

            for i, (xml_file, key_file) in enumerate(izip(xml_files, key_files)):
                if i % 50 == 0:
                    print "Iteration ", i 
                instances = get_instances(xml_file, key_file)
                
                for instance in instances:
                    writer.writerows([instance.get_context_list()])
                    writer.writerows([[instance.label]])
        print 'completed', directory
    except Exception as e:
        logging.exception(e)
        raise e


if __name__ == '__main__':
    assert WORD_TO_INDEX # assert not empty
    #print WORD_TO_INDEX

    pool = Pool(5)
    #for directory in config['directories_for_testing'] :
        #write_csv_for_directory(directory)
    pool.map(write_csv_for_directory, config['directories_for_testing'])
    
    print 'All completed for '
    print config
