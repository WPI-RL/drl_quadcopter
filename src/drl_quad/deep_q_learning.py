import math
import random
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from quadrotor_neuralnetwork import QuadrotorNeuralNetwork
from replay_memory import ReplayMemory

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# Define the state tuple
State = namedtuple('State',
                   ('depth_image', 'relative_position'))

writer = SummaryWriter()


class DeepQLearningAgent:
    def __init__(self, gym_iface,num_episodes=100):
        self.gym_iface = gym_iface
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        self.BATCH_SIZE = 64
        self.GAMMA = 0.5
        self.EPS_START = 1.0
        self.EPS_END = 0.1
        self.TAU = 0.005
        self.LR = 3e-3

        self.num_episodes = num_episodes
        self.num_time_steps = 100000
        self.EPS_DECAY = 0.25
        if self.num_episodes==100:
            self.EPS_DECAY = 25.0 # use for 100 num_episodes
        if self.num_episodes==500:
            self.EPS_DECAY = 105.0 # use for 500 num_episodes
        if self.num_episodes==1000:
            self.EPS_DECAY = 175.0 # use for 1000 num_episodes
        if self.num_episodes==2000:
            self.EPS_DECAY = 300.0 # use for 1000 num_episodes
        
        # writer.add_hparams({"BATCH_SIZE" : self.BATCH_SIZE})
        # writer.add_hparams({"LR" : self.LR})
        # Debug
        self.debug = False

        # Get number of actions from gym action space
        n_actions = self.gym_iface.action_primitives.NUM_ACTIONS

        # Set the model file name
        self.MODEL_FILE_NAME = "quad_dqn_model.pth"

        # Declare the policy and target networks
        n_rel_x = 1
        n_rel_y = 1
        n_rel_z = 1
        self.policy_net = QuadrotorNeuralNetwork(n_rel_x, n_rel_y, n_rel_z, n_actions).to(device)
        # If the model .pth file exists, load it
        try:
            self.policy_net.load_state_dict(torch.load(self.MODEL_FILE_NAME))
            print(f"Model file {self.MODEL_FILE_NAME} found, loading model")
        except:
            print("No model file found, creating new model")
        
        self.target_net = QuadrotorNeuralNetwork(n_rel_x, n_rel_y, n_rel_z, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(100000)

        self.steps_done = 0

        self.num_nn_actions = 0
        self.num_random_actions = 0

        self.no_random_actions = False

    def show_action_stats(self):
        """Shows percentage of actions taken by the neural network and random actions"""
        total_actions = self.num_nn_actions + self.num_random_actions
        print(f"Total actions: {total_actions}")
        print(f"NN actions: {self.num_nn_actions} ({self.num_nn_actions/total_actions*100}%)")
        print(f"Random actions: {self.num_random_actions} ({self.num_random_actions/total_actions*100}%)")
        self.num_nn_actions = 0
        self.num_random_actions = 0

    def select_action(self, state,episode):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * episode / self.EPS_DECAY)
        self.steps_done += 1
        writer.add_scalar("eps_threshold", eps_threshold, episode)
        if self.no_random_actions or sample > eps_threshold:
            self.num_nn_actions += 1
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # if self.debug:
                #     print("self.policy_net(state): ", self.policy_net(state))
                #     print("self.policy_net(state).max(0): ", self.policy_net(state).max(0))
                #     print("self.policy_net(state).max(0).indices: ", self.policy_net(state).max(0).indices)
                #     print("self.policy_net(state).max(0).indices.view(1, 1): ", self.policy_net(state).max(0).indices.view(1, 1))
                action_selected = self.policy_net(state).argmax().view(1, 1)
                return action_selected
        else:
            self.num_random_actions += 1
            # return torch.tensor(
            #     [[random.randrange(self.gym_iface.action_primitives.NUM_ACTIONS)]],
            #     device=device, dtype=torch.long)
            # Choose an action at random from the following set
            # [1, 2, 3, 7, 8, 9, 10, 11, 12, 16, 17]
            #testSet = [1, 2, 3, 7, 8, 9, 10, 11, 12, 16, 17]
            testSet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15, 16, 17]
            return torch.tensor(
                [[random.choice(testSet)]],
                device=device, dtype=torch.long)

    def optimize_model(self,episode):
        if len(self.memory) < self.BATCH_SIZE:
            return
        num_batches = len(self.memory) // self.BATCH_SIZE
        for _ in range(num_batches):
            self.policy_net.optimizer.zero_grad()
            transitions = self.memory.sample(self.BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))
            if self.debug:
                print("batch: ", batch)
                print("batch.state: ", batch.state)
                print("batch.action: ", batch.action)
                print("batch.next_state: ", batch.next_state)

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states_prune = [s for s in batch.next_state if s is not None]
            non_final_next_states = State(
                depth_image=torch.stack([s.depth_image for s in non_final_next_states_prune]),
                relative_position=torch.stack([s.relative_position for s in non_final_next_states_prune])
            )
            state_batch = State(
                depth_image=torch.stack([s.depth_image for s in batch.state]),
                relative_position=torch.stack([s.relative_position for s in batch.state])
            )
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            if self.debug:
                print("state_batch: ", state_batch)
                print("action_batch: ", action_batch)
                print("reward_batch: ", reward_batch)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            q_eval = self.policy_net(state_batch).gather(1, action_batch)
            writer.add_scalar("q_eval", q_eval[0], episode)
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
            # with torch.no_grad():
            next_state_values[non_final_mask] = self.policy_net(non_final_next_states).max(1).values
            
            # Compute the expected Q values
            q_target = (next_state_values * self.GAMMA) + reward_batch
            writer.add_scalar("q_target", q_target[0], episode)
            
            
            #loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            loss = self.policy_net.loss(q_target.unsqueeze(1),q_eval).to(self.policy_net.device)
            writer.add_scalar("Loss/train", loss, episode)
            # Optimize the model
            loss.backward()
            if self.debug:
                print("weight:",self.policy_net.camera_1_layer1.weight)
                print("grad:",self.policy_net.camera_1_layer1.weight.grad)
                print("camera_joint_layer1:",self.policy_net.camera_joint_layer1.weight.grad)
            writer.add_scalar("Highest grad-camera_joint_layer1", self.policy_net.camera_joint_layer1.weight.grad.max().item(), episode)    
            writer.add_scalar("Highest grad-joint_layer1", self.policy_net.joint_layer1.weight.grad.max().item(), episode)    
            self.policy_net.optimizer.step()
            # In-place gradient clipping
            #torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        

    def train(self):
        self.gym_iface.choose_new_goal_position()
        epoch = 0
        epochs_sincelast_optimize=0
        for ep in range(self.num_episodes):
            # if ep % 2 == 0:
            #     self.gym_iface.choose_new_goal_position()
            print(f"\n\n\nDeep-QL Training episode: {ep+1}\n")
            print(f"Goal position: {self.gym_iface.goal_position}\n")
            print(f"Start position: {self.gym_iface.get_current_position()}\n")
            state = State(
                depth_image=self.gym_iface.get_image_set(),
                relative_position=self.gym_iface.get_current_position().unsqueeze(0)
            )
            episodeReward = 0.0
            episode_steps=0
            for _ in range(self.num_time_steps):
                epoch+=1
                epochs_sincelast_optimize+=1
                action = self.select_action(state,ep)
                if self.debug:
                    print("Selected action: ", action)
                # observation, reward, terminated, truncated, _ =
                observation, reward, truncated, terminated = self.gym_iface.step(action.item())
                reward = torch.tensor([reward], device=device)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    # next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    next_state = observation
                
                # if self.debug:
                # if not done:
                #     print("reward: ", reward)
                #     print("next_state: ", next_state)
                # else:
                if self.debug: print("reward: ", reward)
                episodeReward = episodeReward + reward
                episode_steps = episode_steps+1
                # Store the transition in memory
                # state.relative_position = state.relative_position.unsqueeze(0)
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                # target_net_state_dict = self.target_net.state_dict()
                # policy_net_state_dict = self.policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                # self.target_net.load_state_dict(target_net_state_dict)

                
                if done:
                    print("\nEpisode ended due to termination or truncation\n")
                    break
            if epochs_sincelast_optimize > 128:
                epochs_sincelast_optimize = 0
                self.optimize_model(ep)
                # Save the model
                torch.save(self.policy_net.state_dict(), self.MODEL_FILE_NAME)
            writer.add_scalar("episodeReward", episodeReward, ep)
            writer.add_scalar("AverageReward", episodeReward/episode_steps, ep)        
            self.show_action_stats()
        writer.close()

    def run(self):
        self.gym_iface.choose_new_goal_position()
        self.no_random_actions=True
        num_episodes = 10
        num_time_steps = 100
        if torch.cuda.is_available():
            num_episodes = 10
            num_time_steps = 1000
        epoch = 0
        for ep in range(num_episodes):
            # if ep % 2 == 0:
            #     self.gym_iface.choose_new_goal_position()
            print(f"\n\n\nDeep-QL Execution episode: {ep+1}\n")
            print(f"Goal position: {self.gym_iface.goal_position}\n")
            state = State(
                depth_image=self.gym_iface.get_image_set(),
                relative_position=self.gym_iface.get_current_position().unsqueeze(0)
            )
            for _ in range(num_time_steps):
                epoch+=1
                action = self.select_action(state,ep)
                if self.debug:
                    print("Selected action: ", action)
                # observation, reward, terminated, truncated, _ =
                observation, reward, truncated, terminated = self.gym_iface.step(action.item())
                reward = torch.tensor([reward], device=device)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation
                
                if self.debug:
                    print("reward: ", reward)

                # Move to the next state
                state = next_state

                if done:
                    print("\nEpisode ended due to termination or truncation\n")
                    break
            self.show_action_stats()
        writer.close()