import os
import pickle
import numpy as np
import util.util as util

class DataLoader():
        
    def load(self, envname):
        
        filename = os.path.join('expert_data', envname + '.pkl')

        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())
            
            self.observations = data['observations']
            self.actions = np.squeeze(data['actions'])
            self.returns = data['returns']
            self.return_mean = np.mean(self.returns)
            self.return_std = np.std(self.returns)
            assert self.observations.shape[0] == self.actions.shape[0]
            
        print('Environment :', envname)
        print('- num of timesteps :', self.observations.shape[0])
        print('- num of features :', self.observations.shape[1])
        print('- num of actions :', self.actions.shape[1])
        
        # random shuffle traning set
        self.observations, self.actions = \
            util.shuffle_dataset(self.observations, self.actions)
        
        return util.split_dataset(self.observations, self.actions)