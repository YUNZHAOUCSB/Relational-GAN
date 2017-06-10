import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import cPickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension

HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 57 # sequence length
START_TOKEN = 0
#PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
PRE_EPOCH_NUM = 120# supervise (maximum likelihood estimation) epochs
#PRE_EPOCH_NUM = 5# supervise (maximum likelihood estimation) epochs

SEED = 88
BATCH_SIZE = 64
#BATCH_SIZE = 10


#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64

dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
#TOTAL_BATCH = 800
TOTAL_BATCH=200

positive_file = 'converted_equal.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
#generated_num = 10000
generated_num = 659

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)): #downint
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n' #' ' concatenated poem
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0
    #input file
    filename = "/home/yunzhao/PJ/Relation/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    raw_text = open(filename).read().lower().split()
    word_index=sorted(set(raw_text))
    # #embedding
    # embeddings_index = {}
    # f = open('/home/yunzhao/PJ/Relation/glove.840B.300d.txt')
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()

    # print('Found %s word vectors.' % len(embeddings_index))

    # #embedding_matrix
    # embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         embedding_matrix[i] = embedding_vector

    # with tf.device('/cpu:0'),tf.variable_scope('embedding'):
    #     self.W = tf.Variable(embedding_matrix,name='W',dtype=tf.float32)
    #     self.lstm_input = tf.nn.embedding_lookup(self.W, self.input_seq)


    gen_data_loader = Gen_Data_loader(BATCH_SIZE) #64 define an instance
    print gen_data_loader
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing, the same with initial gen_data_load
    vocab_size = 5000
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN) #5000,64,32,32,20,0
    target_params = cPickle.load(open('save/target_params.pkl')) #where does this pkl come from
    #print target_params
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=57, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    #generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)


    gen_data_loader.create_batches(positive_file)
    #print gen_data_loader.token_stream

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            #test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'loss ', loss
            buffer = 'epoch:\t'+ str(epoch) + '\tloss:\t' + str(loss) + '\n'
            log.write(buffer)
    samples1 = generator.generate(sess)
    np.save('generate_file_withoutadversarial.txt',samples1)

    print 'Start pre-training discriminator...'
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(50):#50
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            #test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tloss:\t' + str(loss) + '\n'
            print 'total_batch: ', total_batch, 'loss: ', loss
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

    samples = generator.generate(sess)
    np.save('generate_file.txt',samples)
    print samples
    print type(samples)
    log.close()

# convert to sequences
    word2id={}
    id2word={}
    with open('./Cause_Effect_nosym1.txt','r') as f:
        words = f.read().split()
    for idx,item in enumerate(sorted(set(words))):
        word2id[item] = idx
        id2word[idx] = item 
    word2id[' ']=idx+1
    id2word[idx+1]=' '
    print idx   

    g=open('./generated_seq.txt','w')

    f=np.load('./generate_file.txt.npy')
    for line in f:
        obj=''
        for id in np.array2string(line)[1:-1].split():
            if int(id)>4436:
                id=4436
            obj=obj+id2word[int(id)]+' '
        obj=obj+'\n'
        print obj
        g.write(obj)

if __name__ == '__main__':
    main()
