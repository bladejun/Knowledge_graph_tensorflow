import tensorflow as tf
import numpy as np

from kge_model.model import BaseModel


class TransR(BaseModel):
    def __init__(self, iterator, params):
        super(TransR, self).__init__(iterator, params)
        self.d = params.relation_embedding_dim

    def _score_func(self, h, r, t):
        """ f_r(h,t) = |M_r*h+r-M_r*t| """

        self.Mr = tf.get_variable("Mr", [self.k, self.d], initializer=tf.initializers.identity(gain=0.1))
        self.Mr = tf.tile(tf.expand_dims(self.Mr, 0), [self.batch_size, 1, 1]) # b, k, d
        h = tf.expand_dims(h, axis=1) # b, 1, k
        t = tf.expand_dims(t, axis=1) # b, 1, k
        h_r = tf.squeeze(tf.matmul(h, self.Mr), axis=1)
        t_r = tf.squeeze(tf.matmul(t, self.Mr), axis=1)
        distance = h_r + r - t_r

        if self.params.score_func.lower() == 'l1':  # L1 score
            score = tf.reduce_sum(tf.abs(distance), axis=1)
        elif self.params.score_func.lower() == 'l2':  # L2 score
            score = tf.sqrt(tf.reduce_sum(tf.square(distance), axis=1))
        print(score)
        return score