import tensorflow as tf
import numpy as np

from kge_model.model import BaseModel


class TransH(BaseModel):
    def _score_func(self, h, r, t):
        """f_r(h,t) = |(h-whw)+d_r-(t-wtw)|, w_r,d_r is orthogonal."""
        self.w = tf.get_variable("Mr", [1, 1, self.k])
        self.w = tf.tile(self.w, [self.batch_size, 1, 1])  # (batch_size, 1, embedding_dimension)
        h = tf.expand_dims(h, axis=2)                      # (batch_size, embedding_dimension) -> (batch_size, embedding_dimension, 1)
        t = tf.expand_dims(t, axis=2)                      # (batch_size, embedding_dimension) -> (batch_size, embedding_dimension, 1)
        h_v = tf.squeeze(h, axis=2) - tf.squeeze(tf.matmul(self.w, tf.matmul(h, self.w)), axis=1)
        t_v = tf.squeeze(t, axis=2) - tf.squeeze(tf.matmul(self.w, tf.matmul(t, self.w)), axis=1)
        distance = h_v + r - t_v
        if self.params.score_func.lower() == 'l1':  # L1 score
            score = tf.reduce_sum(tf.abs(distance), axis=1)
        elif self.params.score_func.lower() == 'l2':  # L2 score
            score = tf.sqrt(tf.reduce_sum(tf.square(distance), axis=1))

        return score