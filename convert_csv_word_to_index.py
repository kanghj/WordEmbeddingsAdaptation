import glob


config = {
	'path_to_senna_wordlst': '/home/kanghj/ims_0.9.2/senna/hash/words.lst',			
	}

# maps words to index, this is based on the line number in words.lst
WORD_TO_INDEX = {}


SENSE_TO_INDEX = {}


def construct_word_to_index(wordlst_file):
    for i, word in enumerate(wordlst_file):
        WORD_TO_INDEX[word] = i

with open(config['path_to_senna_wordlst']) as wordlst_file:
    construct_word_to_index(wordlst_file)



if __name__=="__main__":
    if WORD_TO_INDEX.empty():
        raise ValueError('didn\'t initialise correctly')

    files_in_dir = glob.glob('./*.csv')
    for input_file_name in files_in_dir:
        with open('./indexed/' + input_file ) 	as output_file,
             open(input_file_name) 	 	as input_file:
            pass
	    
            



