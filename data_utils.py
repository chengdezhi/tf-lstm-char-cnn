import codecs
import glob
import json
import random

import numpy as np




class Vocabulary(object):

    def __init__(self):
        self._token_to_id = {}
        self._token_to_count = {}
        self._id_to_token = []
        self._num_tokens = 0
        self._s_id = None
        self._unk_id = None
        self._char_to_id = {}
        self._id_to_char = [] 


    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def unk(self):
        return "<UNK>"

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def bos(self):
        return "<S>"

    @property
    def eos(self):
        return "</S>"

    def add(self, token, count):
        self._token_to_id[token] = self._num_tokens
        self._token_to_count[token] = count
        self._id_to_token.append(token)
        self._num_tokens += 1

    def finalize(self):
        self._bos = self.get_id(self.bos)    # <S>
        self._unk_id = self.get_id(self.unk) # <UNK>
        self._eos = self.get_id(self.eos)    # </S>

    def get_id(self, token):
        if token in  self._token_to_id:
            return self._token_to_id.get(token)
        else:
            return self._token_to_id.get(self.unk)
    
    def get_token(self, id_):
        return self._id_to_token[id_]

    def pre_char(self, max_word_length):
        chars_set = set()
        for word in self._id_to_token:
            chars_set |= set(word)
        print "char:", len(chars_set)
        free_ids = []
        for i in range(256):
            if chr(i) in chars_set:
                continue
            free_ids.append(chr(i))
        if len(free_ids) < 4:
            raise ValueError('Not enough free char ids: %d'% len(free_ids))

        self.bow_char = free_ids[0]
        self.eow_char = free_ids[1]
        self.pad_char = free_ids[2]
        self.un_valid_char = free_ids[3]
        chars_set |=  {self.bow_char, self.eow_char, self.pad_char, self.un_valid_char}
        self.chars_len = len(chars_set) 
        for char_id, c in enumerate(chars_set):
            #print "char_id", char_id, c
            self._char_to_id[c] = char_id
            self._id_to_char.append(c)
        
        self._char_set = chars_set
        self._max_word_length = max_word_length
        nums_words = len(self._id_to_token)
        self._word_char_ids = np.zeros([nums_words, max_word_length], dtype=np.int32)

        for i, word in enumerate(self._id_to_token):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self._max_word_length], dtype=np.int32)
        code[:] = ord(self.pad_char)
        if len(word) > self._max_word_length - 2:
            word = word[: self._max_word_length - 2]
        cur_word = self.bow_char + word + self.eow_char
        for j in range(len(cur_word)):
            if cur_word[j] in self._char_to_id:
                code[j] = self._char_to_id[cur_word[j]]
            else:
                code[j] = self._char_to_id[self.un_valid_char]
        return code
    
    def vocab_size(self):
        return len(self._token_to_id), len(self._char_set)


    def get_word_char_ids(self, word):
        if word in self._token_to_id:
            return self._word_char_ids[self._token_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    @staticmethod
    def from_file(filename, vocab_size, word_length):
        vocab = Vocabulary()
        i = 0
        with codecs.open(filename, "r", "utf-8") as f:
            for line in f:
                i += 1
                word, count = line.strip().split()
                vocab.add(word, int(count))
                if i >=vocab_size:
                    break
        vocab.pre_char(word_length)
        vocab.finalize()
        return vocab



class Dataset(object):

    def __init__(self, vocab, file_pattern, deterministic=False):
        self._vocab = vocab
        self._file_pattern = file_pattern
        self._deterministic = deterministic

    def _parse_sentence(self, line):
        s_id = self._vocab._bos 
        e_id = self._vocab._eos
        ids = [s_id] + [self._vocab.get_id(word) for word in line.strip().split()] + [e_id]
        line = self._vocab.bos +  " " + line + " " + self._vocab.eos
        char_ids = [self._vocab.get_word_char_ids(word) for word in line.strip().split()]
        return ids, char_ids

    def _parse_file(self, file_name):
        print("Processing file: %s" % file_name)
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.strip() for line in f]
            if not self._deterministic:
                random.shuffle(lines)
            print("Finished processing!")
            for line in lines:
                yield self._parse_sentence(line)

    def _sentence_stream(self, file_stream):
        for file_name in file_stream:
            for sentence in self._parse_file(file_name):
                yield sentence

    def _iterate(self, sentences, batch_size, num_steps, max_word_length):
        streams = [None] * batch_size
        x = np.zeros([batch_size, num_steps], np.int32)
        x_char = np.zeros([batch_size, num_steps, max_word_length], np.int32)
        y = np.zeros([batch_size, num_steps], np.int32)
        #w = np.zeros([batch_size, num_steps], np.uint8)
        while True:
            x[:] = 0
            y[:] = 0
            x_char[:] = 0
            #w[:] = 0
            for i in range(batch_size):
                tokens_filled = 0
                try:
                    while tokens_filled < num_steps:
                        if streams[i] is None or len(streams[i][0]) <= 1:
                            streams[i] = next(sentences)
                        num_tokens = min(len(streams[i][0]) - 1, num_steps - tokens_filled)
                        #print "stream[i]:",streams[i]
                        #print "pdb:", tokens_filled, num_tokens
                        x[i, tokens_filled:tokens_filled + num_tokens] = streams[i][0][:num_tokens]
                        y[i, tokens_filled:tokens_filled + num_tokens] = streams[i][0][1:num_tokens+1]
                        x_char[i, tokens_filled:tokens_filled + num_tokens] = streams[i][1][:num_tokens]
                        #w[i, tokens_filled:tokens_filled + num_tokens] = 1
                        streams[i][0][:] = streams[i][0][num_tokens:]
                        streams[i][1][:] = streams[i][1][num_tokens:]
                        tokens_filled += num_tokens
                except StopIteration:
                    pass
            yield x, x_char, y

    def iterate_once(self, batch_size, num_steps):
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value

    def iterate_forever(self, batch_size, num_steps, max_word_length):
        def file_stream():
            while True:
                file_patterns = glob.glob(self._file_pattern)
                if not self._deterministic:
                    random.shuffle(file_patterns)
                for file_name in file_patterns:
                    yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps, max_word_length):
            yield value
