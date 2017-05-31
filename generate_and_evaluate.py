from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import time
import numpy as np
import tensorflow as tf

import model
from data_reader import load_data, DataReader, load_data_interface, load_data_for_evaluate_test
from utils import get_topk_index

flags = tf.flags

# datflags.DEFINE_string('load_model',   None,    'filename of the model to load
flags.DEFINE_string('load_model',  'cv/epoch024_4.4120.model',    'filename of the model to load')
flags.DEFINE_string('data_dir',   'data',    'data directory')
flags.DEFINE_integer('num_samples', 1, 'how many words to generate')
flags.DEFINE_float('temperature', 1.0, 'sampling temperature')

# model params
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')

# optimization
flags.DEFINE_integer('max_word_length',     65,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS


def generate_nextwords():
    ''' Loads trained model and evaluates it on test split: next_word.txt '''

    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model + '.meta'):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_data_for_evaluate_test(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS)

    
    print('initialized test dataset reader')
    tw_tensors = word_tensors['for_test']
    tc_tensors = char_tensors['for_test']

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build inference graph '''
        with tf.variable_scope("Model"):
            m = model.inference_graph(
                    char_vocab_size=char_vocab.size,
                    word_vocab_size=word_vocab.size,
                    char_embed_size=FLAGS.char_embed_size,
                    batch_size=1,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    max_word_length=max_word_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    num_unroll_steps=1,
                    dropout=0)

            # we need global step only because we want to read it from the model
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model)
        print('Loaded model from', FLAGS.load_model, 'saved at global step', global_step.eval())

        ''' training starts here '''
        rnn_state = session.run(m.initial_rnn_state)
        logits = np.ones((word_vocab.size,))
        rnn_state = session.run(m.initial_rnn_state)
        

        f = codecs.open(os.path.join('data', 'predict.txt'), 'w', 'utf-8')
        ff = codecs.open(os.path.join('data', 'correct.txt'), 'w', 'utf-8')
        #print(tc_tensors[0,:])
        count = 0
        tlen = len(tw_tensors)
        print("tlen",tlen)
        for j in range(tlen):
            tc_input = np.zeros((1,1,max_word_length))
            tc_input[0,0,:] = tc_tensors[j,:]
            logits, rnn_state = session.run([m.logits, m.final_rnn_state],
                                           {m.input: tc_input,
                                            m.initial_rnn_state: rnn_state})
            logits = np.array(logits)
            # caclulate the accuracy for generate words
            prob = np.exp(logits)
            prob /= np.sum(prob)
            prob = prob.ravel()
            #ix = np.random.choice(range(len(prob)), p=prob)
            ixs = get_topk_index(prob,5)
            w_print = ''
            for ix in ixs:
                w_print += word_vocab.token(ix)+' '
                if j+1<tlen and ix == tw_tensors[j+1]:
                    count += 1
                    word = word_vocab.token(ix)
                    if word == '|':  # EOS
                        word ='<unk>'+' '
                    elif word == '+':
                        word = '\n'
                    else:
                        word = word+' '
                    ff.write(word)
            print("w_print:",w_print)
            word = word_vocab.token(ixs[0])
            if word == '|':  # EOS
                word ='<unk>'+' '
            elif word == '+':
                word = '\n'
            else:
                word = word+' '
            f.write(word)
        print("accuracy:",count*1.0/tlen,"count:",count,"tlen",tlen)
            
        

        genwords = []
        for i in range(FLAGS.num_samples):
            logits = logits / FLAGS.temperature
            prob = np.exp(logits)
            prob /= np.sum(prob)
            prob = prob.ravel()
            ix = np.random.choice(range(len(prob)), p=prob)
            #ix = np.argmax(prob)
            word = word_vocab.token(ix)
            if word == '|':  # EOS
                print('<unk>', end=' ')
            elif word == '+':
                print('\n')
            else:
                print(word, end=' ')
            
            genwords.append(word)
            char_input = np.zeros((1, 1, max_word_length))
            for i,c in enumerate('{' + word + '}'):
                char_input[0,0,i] = char_vocab[c]
            #print("char_input",char_input[0,0,:])
            logits, rnn_state = session.run([m.logits, m.final_rnn_state],
                                         {m.input: char_input,
                                         m.initial_rnn_state: rnn_state})
            logits = np.array(logits)
            if word == '|':  # EOS
                word ='<unk>'+' '
            elif word == '+':
                word = '\n'
            else:
                word = word+' '
        genwords.append(word)
        return genwords
        



        




if __name__ == "__main__":
    result = generate_nextwords()
    #print("result:"+result[0])
