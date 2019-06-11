import os
import tensorflow as tf

class Model():
        
    def __init__(self,
                 num_features, 
                 hidden_list,
                 output_units,
                 learning_rate):
        
        # input placeholder                        
        self.x = tf.placeholder(tf.float32, 
                                shape=(None, num_features), 
                                name='input_x')
        # label placeholder                        
        self.y = tf.placeholder(tf.float32, 
                                shape=(None, output_units),
                                name='output_y')
        # dropout keep probability placeholder                      
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                
        # output, loss, optimizer operation                    
        self.pred = self.network(self.x, hidden_list, output_units, self.keep_prob)
        self.loss = self.get_loss(self.pred, self.y)
        self.optimizer = self.get_optimizer(learning_rate, self.loss)
        
        # loss logging operation for tensorboard
        self.log_training_loss = tf.summary.scalar("training_loss", self.loss)
        self.log_valid_loss = tf.summary.scalar("validation_loss", self.loss)
        self.log_test_loss = tf.summary.scalar("test_loss", self.loss)
        
        # checkpoint saver
        self.saver = tf.train.Saver(var_list=tf.global_variables())

    def network(self, x, hidden_list, output_units, keep_prob):
        # hidden layer
        hidden_activation = tf.nn.leaky_relu
        for units in hidden_list :
            x = tf.layers.dense(inputs=x,
                                units=units,
                                activation=hidden_activation)
            x = tf.layers.dropout(x, keep_prob)

        # output Layer
        x = tf.layers.dense(inputs=x, units=output_units, activation=None)
        return x
        
    def get_loss(self, pred, y):
        # MSE loss
        loss = tf.losses.mean_squared_error(labels=y, predictions=pred)
        return loss

    def get_optimizer(self, learning_rate, loss):
        # Adam optimizer
        optimizer = \
            tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return optimizer


    def update(self, sess, batch_x, batch_y, keep_prob):
        
        # train models
        runs = [self.loss, self.log_training_loss, self.optimizer]
        return sess.run(runs, feed_dict={self.x: batch_x,
                                         self.y: batch_y,
                                         self.keep_prob: keep_prob})
    
    def validate(self, sess, batch_x, batch_y):
        
        # validate models
        runs = [self.loss, self.log_valid_loss]
        return sess.run(runs, feed_dict={self.x: batch_x,
                                         self.y: batch_y,
                                         self.keep_prob: 1})

    def test(self, sess, batch_x, batch_y):
        
        # test models
        runs = [self.loss, self.log_test_loss]
        return sess.run(runs, feed_dict={self.x: batch_x,
                                         self.y: batch_y,
                                         self.keep_prob: 1})

    def predict(self, sess, batch_x):
        
        # predict actions
        return sess.run(self.pred, feed_dict={self.x:batch_x,
                                              self.keep_prob: 1})
 
    def save(self,
             sess,
             checkpoint_dir,
             model_name,
             global_step, 
             write_meta_graph=False):

        # create checkpoint directory
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # save checkpoint file
        self.saver.save(sess, 
                        checkpoint_dir + model_name,
                        global_step)


    def restore(self, sess, checkpoint_dir, restore_file):
        
        # import model graph
        metafile_path = os.path.join(checkpoint_dir, restore_file)
        self.saver = tf.train.import_meta_graph(metafile_path)
        
        # load variables
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            self.saver.restore(sess, latest_checkpoint)