import datetime
import pathlib
import random
import numpy as np
import torch
from .abstract_game import AbstractGame
import random
import tensorflow as tf
import time
import gym
from gym import spaces

# Mesurer le temps d'exécution de MuZero
start_time = time.time()

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (1, 10, 3)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(101))  # Fixed list of all possible actions. You should only edit the length
        #self.action_space = [0, 1, 2, 3, 4,]        
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.02  # Initial learning rate
        self.lr_decay_rate = 0.8  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1.5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    def __init__(self, seed=None):
        num_nodes = 10
        self.num_nodes = num_nodes
        self.nodes = [[1, 0, 0] for _ in range(num_nodes)]  # Initial state: all nodes active, no leader
        self.leader_nodes = []
        self.current_player = 0

    def legal_actions(self):
        actions = []
        for i in range(self.num_nodes):
            if self.nodes[i][0] == 1:  # Node is active
                actions.append(i)
        return actions

    def apply_action(self, action):
        if action not in self.legal_actions():
            raise ValueError("Invalid action")

        self.nodes[action][0] = 0  # Deactivate the selected node

    def evaluate_node(self, node):
        # Criteria values for the node (random values for demonstration)
        latency = random.uniform(0, 1)
        load = random.uniform(0, 1)
        overhead = random.uniform(0, 1)

        # Importance levels for each criteria
        importance = [1, 2, 3]
        #print("latency",latency,"load",load,"overhead",overhead)
        # Calculate evaluation
        evaluation = sum([criteria_value * importance[i] for i, criteria_value in enumerate([latency, load, overhead])])
        

        return evaluation

    def update_evaluations(self):
        for i in range(self.num_nodes):
            self.nodes[i][1] = self.evaluate_node(self.nodes[i])
    def get_observation(self):
        observation = np.expand_dims(np.array(self.nodes), axis=0)
        return observation


    def get_leader(self):
        evaluations = [self.nodes[i][1] for i in range(self.num_nodes)]

        max_evaluation = max(evaluations)
        leaders = [i for i, evaluation in enumerate(evaluations) if evaluation == max_evaluation]

        return leaders

    def step(self, action):
        if action < 0 or action >= self.num_nodes:
            raise ValueError("Invalid action")

        self.apply_action(action)
        self.update_evaluations()

        reward = [1 if i in self.get_leader() else 0 for i in range(self.num_nodes)]
        reward =sum(reward)
        self.leader_nodes = self.get_leader()  # upadate list of leaders

        done = sum(self.nodes[i][0] for i in range(self.num_nodes)) == 1
        obs=self.get_observation()
        #print("obs",obs)
        #print(obs.shape)

        return self.get_observation(), reward, done

    def reset(self):
        self.nodes = [[1, 0, 0] for _ in range(self.num_nodes)]
        self.leader_nodes = []
        observation = self.get_observation()  # Récupérer l'observation après réinitialisation
        return observation

    def render(self):
        print("Cluster Nodes:")
        for i, node in enumerate(self.nodes):
            print(f"Node {i}: {node}")
        print("Leader(s):", self.leader_nodes)

    """def get_observation(self):
        return np.array(self.nodes)"""
# Votre code d'exécution de MuZero ici

end_time = time.time()
execution_time_muzero = end_time - start_time
print("execution time of MuZero :", execution_time_muzero, "secondes")


