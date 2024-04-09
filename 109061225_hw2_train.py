import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gym.wrappers import FrameStack
from gym.spaces import Box
from torchvision import transforms as T
import time

# Setup the environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)


class SkipFrame(gym.Wrapper):
    """
    This class is used to skip frames in the environment.
    It inherits from the gym.Wrapper class and overrides the step() method.
    The step() method repeats the action for a specified number of frames and returns the accumulated reward.
    """

    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    This class is used to convert the RGB observation to grayscale.
    It inherits from the gym.ObservationWrapper class and overrides the observation() method.
    The observation() method converts the RGB observation to grayscale.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """
    This class is used to resize the observation.
    It inherits from the gym.ObservationWrapper class and overrides the observation() method.
    The observation() method resizes the observation to a specified shape.
    """

    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class DQN(nn.Module):
    """mini CNN structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )


# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)


class Agent:
    """
    This class is used to define the agent.
    It contains the following methods:
    - __init__(): This method initializes the agent.
    - preprocess(): This method preprocesses the observation.
    - act(): This method selects an action using the epsilon-greedy policy.
    - cache(): This method adds the experience to memory.
    - recall(): This method samples experiences from memory.
    - td_estimate(): This method calculates the TD estimate.
    - td_target(): This method calculates the TD target.
    - learn(): This method updates the Q_online network.
    - save(): This method saves the model.
    - load(): This method loads the model.
    - update_Q_online(): This method updates the Q_online network.
    - sync_Q_target(): This method synchronizes the Q_target and Q_online networks.
    """

    def __init__(
        self, state_dim=(4, 84, 84), action_dim=env.action_space.n, mode="test"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = mode
        if mode == "test":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = DQN(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        if mode == "test":
            self.exploration_rate = 0.1
            self.exploration_rate_decay = 1
            self.exploration_rate_min = 0.1
            self.load("./109061225_hw2_data")
        else:
            self.exploration_rate = 1
            self.exploration_rate_decay = 0.99999975
            self.exploration_rate_min = 0.1

        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

        self.memory = deque(maxlen=200000)
        self.batch_size = 32

        self.gamma = 0.9

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
        Select action using epsilon-greedy policy
        Parameters
        state : np.ndarray
            current state of the environment
        ----------
        Returns
        action_idx : int
            action index
        """
        # EXPLORE
        p = np.random.rand()
        if p < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = (
                state[0].__array__().copy()
                if isinstance(state, tuple)
                else state.__array__().copy()
            )
            state = torch.tensor(state, device=self.device).float().unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        if self.mode != "test":
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(
                self.exploration_rate_min, self.exploration_rate
            )
        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Add the experience to memory
        Parameters
        state : np.ndarray
            current state of the environment
            next_state : np.ndarray
            next state of the environment
            action : int
            action taken
            reward : float
            reward received after taking action
            done : int
            whether the episode terminated
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append(
            (
                state,
                next_state,
                action,
                reward,
                done,
            )
        )

    def recall(self):
        """Sample experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self, path="./109061225_hw2_data"):
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            path,
        )
        print(f"DQN saved to {path} at step {self.curr_step}")

    def load(self, path="./109061225_hw2_data"):
        save_dict = torch.load(path, map_location=self.device)  # Add map_location here
        self.net.load_state_dict(save_dict["model"])
        self.net = self.net.to(self.device)
        if self.mode != "test":
            self.exploration_rate = save_dict["exploration_rate"]
        print(f"DQN loaded from {path} at step {self.curr_step}")

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    mario = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, mode="train")
    mario.save("./109061225_hw2_data_lite_init")
    episodes = 10000
    for e in range(episodes):
        state = env.reset()
        while True:
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
            state = next_state
            if done or info["flag_get"]:
                print(
                    f'Time: {time.time()} Episode {e} finished after {info["x_pos"]} steps'
                )
                break
        # Save the model every 10 episodes
        if (e + 1) % 10 == 0:
            mario.save(f"./109061225_hw2_data_lite_{e+1}")
            state = env.reset()
            episode_cycle = 0
            total_reward = 0
            while True:
                action = mario.act(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_reward += reward
                episode_cycle += 1
                if done:
                    break
            print(f"Finished after {episode_cycle} cycles, reward:{total_reward}")

    env.close()
