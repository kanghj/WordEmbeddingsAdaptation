import csv
import xml.etree.cElementTree as ET
import codecs
import glob
import os
import logging

import nltk

import multiprocessing
from itertools import izip

"""
This python program converts the one million sense tagged wordnet dataset into csv format. 
This ran on windows. The present program creates a .csv file for every target word. It's possible
to make a csv file for the entire word pos type (adj, adv, noun, verb) by replacing instances of 
write_csv_for_files_in_directory to write_csv_for_directory. 

NLTK is used to produce the part-of-speech (POS) tags. 

(for a csvfile for each target word) Although we are writing a csv file, we don't use any csv headers. 
The first row is number of senses/classes 
for the target word. 

After this, run drop_commas.py. Loading csv files in torch is a pain as of this current date, 
although there is progress getting made by torch in loading csv 
(the csvigo library (which has rtouble loading large files now?)),
so in future, drop_commas may not be required.

"""


config = {
        # convert one million
        #'directories_for_testing' : ['C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/noun', 
        #                              'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/verb',
        #                              'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/adv',
        #                              'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/adj'],
        'directories_for_testing' : ['C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/noun', 
                                      'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/verb',
                                      'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/adv',
                                      'C:/Users/user/Downloads/one-million-sense-tagged-instances-wn30.tar/one-million-sense-tagged-instances-wn30/adj'],


        'wordlst' : 'C:/Users/user/Documents/fyp_embeddings/word2vec2/words.lst',
        }

ENCODING = "iso-8859-1"


# maps words to index, this is based on the line number in words.lst
WORD_TO_INDEX = {}

# map from root word to possible sense id to number
SENSE_TO_INDEX = {}

# penntreebank tags, used by default in nltk pos_tags
# a special 'padding' tag is used for PADDING
POS_TAGS = ['PRP$', 'VBG', 'FW', 'VBN', 'POS', "''", 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', '#', 'RP', '$', 'NN', ')', '(',\
            'VBD', ',', '.', 'TO', 'PRP', 'RB', ':', 'NNS', 'NNP', '``', 'WRB', 'CC', 'LS', 'PDT', 'RBS', 'RBR', 'CD', 'EX',\
            'IN', 'WP$', 'MD', 'NNPS', 'JJS', 'JJR', 'SYM', 'VB', 'UH'] + ['PADDING']


def construct_word_to_index(wordlst_file):
    for i, word in enumerate(wordlst_file):
        WORD_TO_INDEX[word.strip()] = i + 1 # + 1 to be 1-indexed, this is convenient for later use 
                                            # because the lua torch library is 1-indexed
    if 'UNKNOWN' not in WORD_TO_INDEX:
        WORD_TO_INDEX['UNKNOWN'] = len(WORD_TO_INDEX) 

with open(config['wordlst']) as wordlst_file:
    print "gathering WORD_TO_INDEX mapping"
    construct_word_to_index(wordlst_file)

def get_root_word(senseid):
    return senseid.split('%')[0]

def construct_or_recover_sense_to_index():
    
    def construct_sense_to_index_from_keyfile(opened_file):
        for line in opened_file:
            senseid = line.split(' ')[2].strip()
            root_word = get_root_word(senseid)
            
            # lookup root word in SENSE TO INDEX
            try:
                senses = SENSE_TO_INDEX[root_word]
            except KeyError:
                senses = {}
                SENSE_TO_INDEX[root_word] = senses

            try:
                sense_number = senses[senseid]
            except KeyError:
                senses[senseid] = len(senses) + 1

    def recover_sense_to_index(opened_file):
        for line in opened_file:
            root_word = get_root_word(line.split()[0])
            senseid = line.split()[0]
            sense_number = line.split()[1]

            try:
                SENSE_TO_INDEX[root_word]
            except:
                SENSE_TO_INDEX[root_word] = {}

            SENSE_TO_INDEX[root_word][senseid] = sense_number

    def save_sense_to_index_to_file(opened_file):
        for root_word, value in SENSE_TO_INDEX.iteritems():
            for senseid, sense_number in value.iteritems():
                opened_file.write(senseid + ' ' + str(sense_number) + '\n')

    print "gathering SENSE_TO_INDEX mapping"
    if os.path.isfile('./senseid_to_index') and os.path.getsize('./senseid_to_index') > 0:
        key_file = open('./senseid_to_index', 'r')
        recover_sense_to_index(key_file)
        key_file.close()
    else:
        for directory in config['directories_for_testing']:
            key_files = glob.glob(directory + '/*.key')
            for key_file_name in key_files:
                key_file = open(key_file_name, 'r')

                construct_sense_to_index_from_keyfile(key_file)

                key_file.close()

        saved_file = open('./senseid_to_index', 'w+')
        save_sense_to_index_to_file(saved_file)
        saved_file.close()

construct_or_recover_sense_to_index()

class Instance(object):
    number = -1
    head = ""
    tail = ""
    label = ""
    label_as_int = 0

    head_pos_tags = []
    tail_pos_tags = []

    training_instance_strategy = "window"

    @classmethod
    def _get_context(cls, head, tail):
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
            front_list = head.split()[-5 :]
            back_list = tail.split()[0 : 5]
            front = ' '.join(front_list)
            back = ' '.join(back_list)

            if len(front_list) < 5:
                padding_front = ["PADDING"] * (5-len(front_list))
                front = ' '.join(padding_front) + ' ' + front
            if len(back_list) < 5:
                padding_back = ["PADDING"] * (5-len(back_list))
                back = back + ' ' + ' '.join(padding_back)
        else:
            raise ValueError("invalid training strategy : " + \
                              cls.training_instance_strategy)
        return (front + ' ' + back).encode(ENCODING)

    
    def _get_postags(self, head, tail):
        if self.training_instance_strategy == "raw":
            return self.pos_tags
        elif self.training_instance_strategy == "sentence":
            raise Exception('not implemented yet')
            
        elif self.training_instance_strategy == "window":
            front = self.head_pos_tags[-5 :]
            back = self.tail_pos_tags[0 : 5]
            return front + back
        else:
            raise ValueError("invalid training strategy : " + \
                              self.training_instance_strategy)
        

    def __init__(self, number, head, tail, head_pos_tags, tail_pos_tags, label):
        self.number = number
        self.head = ' '.join(head.replace('\n','').split()).strip()
        self.tail = ' '.join(tail.replace('\n','').split()).strip()
        self.label = label.replace('\n','').strip()
        self.head_pos_tags = head_pos_tags
        if len(self.head_pos_tags) < 5:
            padding_front = ["PADDING"] * (5-len(self.head_pos_tags))
            self.head_pos_tags = padding_front + self.head_pos_tags

        self.tail_pos_tags = tail_pos_tags
        if len(self.tail_pos_tags) < 5:
            padding_back = ["PADDING"] * (5-len(self.tail_pos_tags))
            self.tail_pos_tags =  self.tail_pos_tags + padding_back


    def __repr__(self):
        return "Instance[id:" + str(self.number) + \
               ", context: " + self.head + " " + self.tail + ", answer:" + \
                self.label + "]"
    
    def get_context_list(self):
        context_words = Instance._get_context(self.head, self.tail)
        context_indices = []
        
        assert len(context_words.split()) == 10, "context length wrong? word is " + str(self.number) + " , " + str(context_words) + " : " + str(context_words.split())

        for i, word in enumerate(context_words.split()):
            try:
                context_indices.append(WORD_TO_INDEX[word.lower()])
            except KeyError:
                context_indices.append(WORD_TO_INDEX['UNKNOWN'])
        return context_indices

    def get_pos_tags_list(self):
        pos_tags = self._get_postags(self.head, self.tail)
        pos_indices = []
        
        for i, word in enumerate(pos_tags):
            try:
                pos_indices.append(POS_TAGS.index(word))
            except ValueError:
                pos_indices.append(-1) # append -1 so that in the end, pos_indices will have the right length
                logging.warning('.... incorrect data during POS tagging')

        return pos_indices


def get_instances(xml_file, key_file):
    def should_be_omitted(num_instances):
        return num_instances < 5


    tree = ET.parse(xml_file)
    xml_instances = tree.getroot().findall('.//instance')

    ids     = map(lambda x: x.attrib['id'], xml_instances)
    if should_be_omitted(len(ids)):
        return []

    heads   = map(lambda x: x.find('context').text or '', xml_instances)
    # the part of the context behind the <head> doesn't get included in head
    # so we use the tail of the head to obtain it
    tails   = map(lambda x: x.find('.//head').tail or '', xml_instances)

    full_context = [(head + tail).split() for head, tail in izip(heads, tails)]
    pos_tags_of_full_context = nltk.pos_tag_sents(full_context)

    with open(key_file) as labels_file:
        labels = [line.split(' ')[2] for line in labels_file]
        # labels is now the sense in wordnet, in the senseval format
        # but we should convert it into a numbered format, based on a key file (use SENSE_TO_INDEX)
    
    csv_instances = []
    for number, head, tail, label, pos_tags in izip(ids, heads, tails, labels, pos_tags_of_full_context):
        head_pos_tags = pos_tags[: len(head.split())]
        head_pos_tags = map(lambda x : x[1], head_pos_tags)

        assert len(head_pos_tags) == len(head.split())

        tail_pos_tags = pos_tags[len(head.split()) :]
        tail_pos_tags = map(lambda x : x[1], tail_pos_tags)

        assert len(tail_pos_tags) == len(tail.split())

        instance = Instance(number, head, tail, head_pos_tags, tail_pos_tags, label)

        csv_instances.append(instance)

    return csv_instances


def write_csv_for_directory(directory):
    try:
        with open('./' + directory.split('/')[-1] + '.csv', 'wb+' )  as output_file:
            writer = csv.writer(output_file)

            # input files are *.xml and *.key, *.key contains the answers, *.xml contains instance,context info
            xml_files = glob.glob(directory + '/*.xml')
            key_files = glob.glob(directory + '/*.key')

            for i, (xml_file, key_file) in enumerate(izip(xml_files, key_files)):
                if i % 50 == 0:
                    print multiprocessing.current_process(), "Iteration ", i 
                instances = get_instances(xml_file, key_file)
                
                for instance in instances:
                    row = instance.get_context_list()
                    row.extend(instance.get_pos_tags_list())
                    writer.writerows([row])
                    writer.writerows([[SENSE_TO_INDEX[ get_root_word(instance.label)][instance.label] ]])
        print 'completed', directory
    except Exception as e:
        logging.exception(e)
        print multiprocessing.current_process()
        raise e


def write_csv_for_files_in_directory(directory):
    """
    Writes to csv file in directory './testfiles/' + word_type + file_name + '.csv'
    """
    try:
        word_type = directory.split('/')[-1]
        xml_files = glob.glob(directory + '/*.xml')
        key_files = glob.glob(directory + '/*.key')

        for i, (xml_file, key_file) in enumerate(izip(xml_files, key_files)):
            if i % 50 == 0:
                print multiprocessing.current_process(), "Iteration ", i 
            
            file_name = xml_file.split('\\')[-1].split('/')[-1].split('.')[0]  # split on both \ and / in case of os differences (may not matter)
                                                                                # this is done in order to fix some strange bug
                                                                                # note: I ran this code on windows, unix is untested
            instances = get_instances(xml_file, key_file)
            if instances:
                with open('./testfiles/' + word_type + file_name + '.csv', 'wb+')  as output_file:
                    writer = csv.writer(output_file)
                    
                    # write number of classes first
                    writer.writerow([len(SENSE_TO_INDEX[get_root_word(instances[0].label)])])

                    for instance in instances:
                        writer.writerows([instance.get_context_list() + instance.get_pos_tags_list()])
                        writer.writerows([[SENSE_TO_INDEX[ get_root_word(instance.label)][instance.label] ]])

        print 'completed', directory
    except Exception as e:
        logging.exception(e)
        print multiprocessing.current_process()
        raise e


if __name__ == '__main__':
    assert WORD_TO_INDEX # assert not empty
    assert POS_TAGS
    assert SENSE_TO_INDEX


    pool = multiprocessing.Pool(processes=5)
    #for directory in config['directories_for_testing'] :
    #    write_csv_for_files_in_directory(directory)
    pool.map(write_csv_for_files_in_directory, config['directories_for_testing'])
    
    print 'All completed for '
    print config