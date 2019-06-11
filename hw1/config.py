"""config.py.

This module includes Config class.

"""
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".
from util.util import ModelInit
from util.util import ImitationMode

class EasyDict(dict):
    """Custom dictionary class for configuration."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        """Get attribute."""
        return self[name]

    def __setattr__(self, name, value):
        """Set attribute."""
        self[name] = value

    def __delattr__(self, name):
        """Delete attribute."""
        del self[name]

# python run_expert.py experts/Ant-v2.pkl Ant-v2 --render --num_rollouts 20
class Config():
    """Configuration Class."""

    def __init__(self):
        """Initialize all configuration variables."""
        
        # Environment Parameters
        self.environment = EasyDict()
        self.environment.envlist = ['Ant-v2',
                                    'HalfCheetah-v2',
                                    'Hopper-v2',
                                    'Humanoid-v2',
                                    'Reacher-v2',
                                    'Walker2d-v2']
        self.environment.rolloutlist = [250, 250, 250, 250, 250, 250]
        self.environment.max_timesteps = 5000
        
        # Model Parameters
        self.model = EasyDict()
        self.model.hidden_list = {'Ant-v2': [64, 32], # 11 obs, 8 actions
                                  'HalfCheetah-v2': [64, 32], # 17 obs, 6 actions
                                  'Hopper-v2': [64, 32], # 11 obs, 3 actions
                                  'Humanoid-v2': [200, 100], # 376 obs, 17 actions
                                  'Reacher-v2': [64, 32], # 11 obs, 2 actions
                                  'Walker2d-v2': [64, 32]} # 11 obs, 2 actions

        # Behavior Cloning Parameters
        self.bc = EasyDict()
        self.bc.envname = 'Walker2d-v2'
        self.bc.imitation_mode = ImitationMode.bc # bc, DAgger

        
        # train policy
        self.bc.epochs = 30
        self.bc.max_epochs = 200
        self.bc.batch_size = 64

        self.bc.learning_rate = 3e-4
        self.bc.keep_prob = 0.5
        self.bc.display_step = 500
        
        # test policy
        self.bc.test_display_step = 100
        self.bc.num_rollouts = 10
        self.bc.max_steps = 3000

        # early stop
        self.bc.early_stop_threshold = 51e-5
        self.bc.early_stop_count_threshold = 10
        self.bc.loss_convergence_threshold = 5e-3
        
        # checkpoint save/restore
        self.bc.checkpoint_step = 10000
        self.bc.checkpoint_dir = './ckpt/'
        self.bc.restore = ModelInit.new # new, restore_train, restore_test
        self.bc.restore_file = 'Walker2d-v2-25001.meta'
        
        # tensorboard file writer
        self.bc.log_dir = './logs/'

