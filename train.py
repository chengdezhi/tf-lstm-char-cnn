from __future__ import absolute_import
from __future__ import division

import os
import time
import numpy as np
import tensorflow as tf
import model
from data_utils import Vocabulary, Dataset
from common import load_from_checkpoint
flags = tf.flags

# data
flags.DEFINE_string('vocab_file', '/data1/cdz/1b_data/lm_vocab.txt', 'Vocabulary file.')
flags.DEFINE_string('train_data', '/data1/cdz/1b_data/train/*',    
                                  'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('valid_data', '/data1/cdz/1b_data/heldout/*',    
                                  'data directory. Should contain valid.txt  with input data')
# vocab size
flags.DEFINE_string('vocab_size', 100000, 'specific vocab size')


flags.DEFINE_string('train_dir',   'cv',     'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   None,    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size',        300,                              'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')

# optimization
flags.DEFINE_float  ('learning_rate_decay', 0.95,  'learning rate decay')
flags.DEFINE_float  ('learning_rate',       0.5,  'starting learning rate')
flags.DEFINE_float  ('decay_when',          1.0,  'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float  ('param_init',          0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps',    35,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          256,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs',          200,   'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('max_word_length',     50,   'maximum word length')

flags.DEFINE_integer('num_sampled',         8196, 'sampled for softmax')
flags.DEFINE_integer('arg_max',             5 ,   'get_top_k_of_logits')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_integer('print_every',    30,    'how often to print current loss')
FLAGS = flags.FLAGS


def test():
    vocab = Vocabulary.from_file(FLAGS.vocab_file, FLAGS.vocab_size, FLAGS.max_word_length)
    v_dataset = Dataset(vocab, FLAGS.valid_data)
    # Use only 4 threads for the evaluation.                                                
    config = tf.ConfigProto(allow_soft_placement=True,                                      
                        intra_op_parallelism_threads=20,                                
                        inter_op_parallelism_threads=1)                                 
    config.gpu_options.allow_growth=True
    with tf.Graph().as_default():
        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)
        ''' build training graph '''
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
         
        ''' build graph for validation (shares parameters with the training graph!) '''
        with tf.variable_scope("Model"):
            valid_model = model.inference_graph(
                    char_vocab_size=vocab.chars_len,
                    word_vocab_size=FLAGS.vocab_size,
                    char_embed_size=FLAGS.char_embed_size,
                    batch_size=1,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    max_word_length=FLAGS.max_word_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    num_unroll_steps=FLAGS.num_unroll_steps,
                    dropout=0.0,
                    arg_max=5,
                    mode="valid")
        v_data_iter =  v_dataset.iterate_forever(1, FLAGS.num_unroll_steps, FLAGS.max_word_length)
        sw = tf.summary.FileWriter("cv", tf.Graph())
        saver = tf.train.Saver()
        with tf.Session(config=config).as_default() as sess:
            load_from_checkpoint(saver,"cv")
            # for dev evaluate:
            valid_rnn_state = sess.run(valid_model.initial_rnn_state)
            avg_valid_loss = 0.0
            valid_cnt = 41
            for count in range(valid_cnt):
                x_word, x_char, y = next(v_data_iter)
                loss, valid_rnn_state = sess.run([
                    valid_model.loss,
                    valid_model.final_rnn_state
                ], {
                    valid_model.input  : x_char,
                    valid_model.targets: y,
                    valid_model.initial_rnn_state: valid_rnn_state
                })
                if count % FLAGS.print_every == 0:
                    print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
                avg_valid_loss  += loss/valid_cnt
    pass

def main(_):
    
    vocab = Vocabulary.from_file(FLAGS.vocab_file, FLAGS.vocab_size, FLAGS.max_word_length)
    t_dataset = Dataset(vocab, FLAGS.train_data)
    v_dataset = Dataset(vocab, FLAGS.valid_data)
    print('initialized all dataset readers')

    # Use only 4 threads for the evaluation.                                                
    config = tf.ConfigProto(allow_soft_placement=True,                                      
                        intra_op_parallelism_threads=20,                                
                        inter_op_parallelism_threads=1)                                 
    config.gpu_options.allow_growth=True
    with tf.Graph().as_default():
        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)
        ''' build training graph '''
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("Model", initializer=initializer):
            train_model = model.inference_graph(
                    char_vocab_size=vocab.chars_len,
                    word_vocab_size=FLAGS.vocab_size,
                    char_embed_size=FLAGS.char_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    max_word_length=FLAGS.max_word_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    num_unroll_steps=FLAGS.num_unroll_steps,
                    dropout=FLAGS.dropout,
                    num_sampled=FLAGS.num_sampled,
                    mode="train")
            #train_model.update(model.loss_graph(train_model.logits, FLAGS.batch_size, FLAGS.num_unroll_steps, "train"))

            # scaling loss by FLAGS.num_unroll_steps effectively scales gradients by the same factor.
            # we need it to reproduce how the original Torch code optimizes. Without this, our gradients will be
            # much smaller (i.e. 35 times smaller) and to get system to learn we'd have to scale learning rate and max_grad_norm appropriately.
            # Thus, scaling gradients so that this trainer is exactly compatible with the original
            train_model.update(model.training_graph(train_model.loss * FLAGS.num_unroll_steps,
                    FLAGS.learning_rate, FLAGS.max_grad_norm))

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        # saver = tf.train.Saver(max_to_keep=50)

        ''' build graph for validation (shares parameters with the training graph!) '''
        with tf.variable_scope("Model", reuse=True):
            valid_model = model.inference_graph(
                    char_vocab_size=vocab.chars_len,
                    word_vocab_size=FLAGS.vocab_size,
                    char_embed_size=FLAGS.char_embed_size,
                    batch_size=1,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    max_word_length=FLAGS.max_word_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    num_unroll_steps=FLAGS.num_unroll_steps,
                    dropout=0.0,
                    arg_max=5,
                    mode="valid")
        
        sw = tf.summary.FileWriter("cv", tf.Graph())
        best_valid_loss = None
        avg_train_loss = 0.0
        sv = tf.train.Supervisor(is_chief=True,
                        logdir="cv",
                        summary_op=None,  # Automatic summaries don't work with placeholders.
                        global_step=train_model.global_step,
                        save_summaries_secs=30,
                        save_model_secs=120 * 5)
        with sv.managed_session("", config=config) as sess:
            # Slowly increase the number of workers during beginning of the training.          
            # sess.run(tf.assign(train_model.learning_rate, FLAGS.learning_rate))
            rnn_state = sess.run(train_model.initial_rnn_state)
            local_step =-1                                                                                    
            prev_global_step = sess.run(train_model.global_step)                                                   
            prev_time = time.time()                                                                          
            t_data_iter =  t_dataset.iterate_forever(FLAGS.batch_size, FLAGS.num_unroll_steps, FLAGS.max_word_length)
            v_data_iter =  v_dataset.iterate_forever(1, FLAGS.num_unroll_steps, FLAGS.max_word_length)
            current_learning_rate = FLAGS.learning_rate 
            while not sv.should_stop():
                fetches = [train_model.global_step, train_model.loss, train_model.train_op, train_model.final_rnn_state]                                    
                # for train
                x_word, x_char, y  = next(t_data_iter)                                                              
                # print "feed:", x.shape, y.shape, rnn_state.shape 
                fetched = sess.run(fetches,{train_model.input: x_char, train_model.targets: y, train_model.initial_rnn_state: rnn_state, train_model.learning_rate: current_learning_rate})                        
                rnn_state = fetched[-1]          
                local_step += 1                                                                              
                if local_step < 10 or local_step % 20 == 0:                                                  
                    cur_time = time.time()                                                                   
                    num_words = FLAGS.batch_size * FLAGS.num_unroll_steps                                
                    wps = (fetched[0] - prev_global_step) * num_words / (cur_time - prev_time)               
                    prev_global_step = fetched[0]                                                            
                    
                    print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f, ppl = %.4f" % (        
                        fetched[0], cur_time - prev_time, wps, fetched[1], np.exp(fetched[1])))              
                    prev_time = cur_time   
                # for dev evaluate:
                valid_rnn_state = sess.run(valid_model.initial_rnn_state)
                if local_step % 100 == 0 :
                    valid_cnt =  512 
                    avg_valid_loss = 0.0
                    for count in range(valid_cnt):
                        x_word, x_char, y = next(v_data_iter)
                        loss, valid_rnn_state = sess.run([
                            valid_model.loss,
                            valid_model.final_rnn_state
                        ], {
                            valid_model.input  : x_char,
                            valid_model.targets: y,
                            valid_model.initial_rnn_state: valid_rnn_state
                        })
                        if count % FLAGS.print_every == 0:
                            print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
                        avg_valid_loss  += loss/valid_cnt
                    print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))

                    ''' write out summary events '''
                    summary = tf.Summary(value=[
                    tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss)
                    ])
                    sw.add_summary(summary, fetched[0])

                    ''' decide if need to decay learning rate '''
                    if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
                        print('validation perplexity did not improve enough, decay learning rate')
                        print('learning rate was:', current_learning_rate)
                        current_learning_rate *= FLAGS.learning_rate_decay
                        if current_learning_rate < 1.e-5:
                            print('learning rate too small - stopping now')
                            break
                        print('new learning rate is:', current_learning_rate)
                    else:
                        best_valid_loss = avg_valid_loss
        sv.stop()                                                                                            



if __name__ == "__main__":
    #test()
    tf.app.run()
    #run_test()
