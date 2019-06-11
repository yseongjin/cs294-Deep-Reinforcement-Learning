import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import config
import gym

from util.data_loader import DataLoader
from model.model import Model
from util.util import ModelInit
import util.util as util

class BehaviorCloning():
    
    def __init__(self, config):
        self.config = config
        self.envname = config.bc.envname

    def train(self):
        
        # Hyper parameters
        epochs = self.config.bc.epochs
        batch_size = self.config.bc.batch_size
        display_step = self.config.bc.display_step
        keep_prob = self.config.bc.keep_prob
        
        # check point configrations
        checkpoint_step = self.config.bc.checkpoint_step
        checkpoint_dir = self.config.bc.checkpoint_dir
        
        # load expert data
        expert_data_loader = DataLoader()
        
        self.x_train, self.y_train, \
        self.x_valid, self.y_valid, \
        self.x_test, self.y_test = expert_data_loader.load(self.envname)      

        # calculate the number of neuraon for each layer
        num_timesteps, num_features =  self.x_train.shape
        num_actions =  self.y_train.shape[1]

        # Training Phase
        print('Training...')
        with tf.Session() as sess:
            # tensorboard logger
            self.tb_logger = self.get_tb_logger(sess, self.envname)

            # build model
            self.model, continue_train = \
                self.create_model(sess, num_features, num_actions)
            
            # if not necessary to train the model, do test
            if not continue_train: 
                self.test(sess, expert_data_loader)
                return
            
            # Training cycle
            num_steps = num_timesteps // batch_size
            loss_list = []
            v_loss_list = []
            last_v_loss = 0
            early_stop = self.reset_early_stop()
            global_step = 1
            epoch = 1

            while epoch <= epochs:
                print('epoch ', epoch)
                self.x_train, self.y_train = \
                    util.shuffle_dataset(self.x_train, self.y_train)
                
                # mini batch iterations
                for step in range(0, num_steps):
                    start_idx = step*batch_size
                    end_idx = (step+1)*batch_size
                    x_obs = self.x_train[start_idx:end_idx]
                    x_actions = self.y_train[start_idx:end_idx]
                    
                    loss, log_loss, _ = \
                        self.model.update(sess, x_obs, x_actions, keep_prob)
                    if loss == 0 : print(step, "loss is zero")
                    if global_step % display_step == 0:
                        # validation
                        v_loss, log_v_loss = self.model.validate(sess,
                                                                 self.x_valid,
                                                                 self.y_valid)

                        # early stopping
                        early_stop = self.check_early_stop(v_loss, last_v_loss)

                        print("step " + str(global_step) + \
                              ", train loss " + "{:.5f}".format(loss) + \
                              ", validation loss " + "{:.5f}".format(v_loss))
                        
                        # tensorboard logging
                        self.tb_logger.add_summary(log_loss, global_step)  
                        self.tb_logger.add_summary(log_v_loss, global_step)  

                        # make loss list for plotting
                        last_v_loss = v_loss
                        loss_list.append(loss)
                        v_loss_list.append(v_loss)
                        
                    # Save Model
                    if global_step % checkpoint_step == 0:
                        self.model.save(sess,
                                        checkpoint_dir,
                                        self.envname,
                                        global_step)
                    if early_stop : break
                    global_step += 1
                
                if early_stop : break
                epoch += 1
                # if loss is greater than the threshold, increase # epochs
                epochs = self.check_epochs(epochs, epoch, loss)

            print("step " + str(global_step) + \
                  ", train loss " + "{:.5f}".format(loss) + \
                  ", validation loss " + "{:.5f}".format(v_loss))
                        
            # Save Model
            self.model.save(sess,
                            checkpoint_dir,
                            self.envname,
                            global_step,
                            True)

            # show loss plot
            self.show_train_graph(loss_list, v_loss_list)

            # test policy
            self.test(sess, expert_data_loader)

    def create_model(self, sess, num_features, num_actions):
        
        # model configuratoin
        learning_rate = self.config.bc.learning_rate
        hidden_list = self.config.model.hidden_list[self.envname]
        
        # create a model
        model =  Model(num_features,
                       hidden_list,
                       num_actions, 
                       learning_rate)
        continue_train = True

        # check point configuratoin
        checkpoint_dir = self.config.bc.checkpoint_dir
        restore = self.config.bc.restore
        restore_file = self.config.bc.restore_file
       
        
        # initialize or restore the model
        if restore == ModelInit.new:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())
            
        elif restore == ModelInit.restore_test:
            model.restore(sess, checkpoint_dir, restore_file)
            continue_train = False
            
        elif restore == ModelInit.restore_train:
            model.restore(sess, checkpoint_dir, restore_file)
            # need to develop training from restored time steps
        
        return model, continue_train
            
    def show_train_graph(self, loss_list, v_loss_list):
        plt.plot(loss_list)
        plt.plot(v_loss_list)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.show()

    def summary_returns(self, returns, title):
        time_steps = returns.shape[0]
        return_mean = np.mean(returns)
        return_std = np.std(returns)
           
        print() 
        print(title, " Return Summary:") 
        print("Rollouts : ", time_steps)
        print("Mean : ",  return_mean)
        print("Stdev : ", return_std)

    def reset_early_stop(self):
        self.early_stop_count = 0
        return False
        
    def check_early_stop(self, loss, last_loss):
        early_stop_threshold = self.config.bc.early_stop_threshold 
        early_stop_count_threshold = self.config.bc.early_stop_count_threshold 
        diff = loss - last_loss
        
        if abs(diff) < early_stop_threshold:
            self.early_stop_count +=1

            if self.early_stop_count >= early_stop_count_threshold:
                print("v_loss - last_v_loss ", diff,
                      "early_stop_count", self.early_stop_count)
                return True
            
        return self.reset_early_stop()     
        
    def check_epochs(self, epochs, epoch, loss):
        
        threshold = self.config.bc.loss_convergence_threshold
        if epoch > epochs and abs(loss) > threshold:
            max_epochs = self.config.bc.max_epochs
            new_epochs = epochs + int(epochs* 0.1)
            return min([new_epochs, max_epochs])
            
        return epochs

    def get_tb_logger(self, sess, envname, bc_type='bc'):

        log_dir = self.config.bc.log_dir
        log_path = os.path.join(log_dir, envname, bc_type)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            
        return tf.summary.FileWriter(log_path, sess.graph)
    
    def test(self, sess, expert_data_loader):
        self.test_policy(sess)
        
        # rollout bc policy
        num_rollouts = self.config.bc.num_rollouts
        max_steps = self.config.bc.max_steps
        experience = self.rollout_bc_policy(sess,
                                                 self.envname,
                                                 max_steps,
                                                 num_rollouts)
        
        self.summary_returns(expert_data_loader.returns, "Expert")
        self.summary_returns(experience['returns'], "Behavior Cloning")

    def test_policy(self, sess):
        print('Testing...')

        num_timesteps =  self.x_test.shape[0]
        batch_size = self.config.bc.batch_size
        display_step = self.config.bc.test_display_step
        
        num_steps = num_timesteps // batch_size
        loss_list = []

        # mini batch iterations
        for step in range(1, num_steps+1):
            start_idx = step*batch_size
            end_idx = (step+1)*batch_size
            x_obs = self.x_test[start_idx:end_idx]
            x_actions = self.y_test[start_idx:end_idx]
           
            if step % display_step == 1:
                # Calculate batch loss and accuracy
                loss, log_test_loss = self.model.test(sess, x_obs, x_actions)

                print("step " + str(step) + \
                      ", test loss " + "{:.5f}".format(loss))

                self.tb_logger.add_summary(log_test_loss, step)  
                loss_list.append(loss)
            
    def rollout_bc_policy(self, sess, envname, max_steps, num_rollouts=10, render=True):
        observations = []
        actions = []
        returns = []
        
        env = gym.make(envname)
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                observations.append(obs)
                actions.append(self.model.predict(sess, np.expand_dims(obs,axis=0))[0])
                obs, r, done, _ = env.step(actions[-1])
                totalr += r
                steps += 1
                if render: env.render()
                if steps >= max_steps : break
            
            returns.append(totalr)

        experience = {'observations': np.array(observations),
                      'actions': np.array(np.squeeze(actions)),
                      'returns':np.array(returns)}
        return experience

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    
    cfg = config.Config()
    
    # read expert data
    bc = BehaviorCloning(cfg)
    policy_fn = bc.train()

    end_time = datetime.datetime.now()           
    print()
    print("Exiting...", end_time)
    print("Running Time", end_time - begin_time)