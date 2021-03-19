import codecs
import tensorflow as tf


def print_args(flags):
    print("\n Parameters:")
    for attr in flags:
        value = flags[attr].value
        print("{%s}={%s}" % (attr, value))
    print("")


def load_vocab(vocab_file):
    vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocab_file, default_value=0)
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())

    return vocab_table, vocab, vocab_size

# restore checkpoint model
def load_model(sess, ckpt):
    with sess.as_default():
        with sess.graph.as_default():
            init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]
            sess.run(init_ops)
            ckpt_path = tf.train.latest_checkpoint(ckpt)
            if ckpt_path:
                print("Loading saved model: %s" % ckpt_path)
            else:
                raise ValueError("No checkpoint found in %s" % ckpt)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_path)