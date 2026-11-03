"""
agent.py -- the tabular Q-learning AGENT for Project03_RL.

The agent knows nothing about Flappy Bird. It only knows:
  - it receives a state (three numbers)
  - it can pick action 0 (do nothing) or 1 (flap)
  - afterwards it receives a reward
From those alone it builds a Q-table: for every (state, action) pair, an
estimate of the total future reward it can expect.
"""

import pickle
import random
from collections import defaultdict

N_ACTIONS = 2


class QAgent:
    def __init__(self,
                 alpha=0.7,             # learning rate: how far each update moves Q
                 gamma=0.95,            # discount: how much future reward matters
                 epsilon=0.1,           # starting chance of a random (exploratory) action
                 epsilon_min=0.0,
                 epsilon_decay=0.995,   # epsilon shrinks by this factor each episode
                 dx_bucket=20,          # bucket sizes for discretizing the state
                 dy_bucket=15,
                 vel_bucket=2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.dx_bucket = dx_bucket
        self.dy_bucket = dy_bucket
        self.vel_bucket = vel_bucket

        # Q-table: maps a discretized state -> [Q(nothing), Q(flap)].
        # defaultdict means unseen states start at [0.0, 0.0].
        self.q = defaultdict(lambda: [0.0] * N_ACTIONS)

    # ---------------- State discretization ----------------
    def discretize(self, state):
        """
        Turn the raw (dx, dy, vel) floats into a small tuple of integers.
        A table can only store a finite number of states, so nearby
        situations are grouped into the same 'bucket'.
        """
        dx, dy, vel = state
        dx = min(int(dx // self.dx_bucket), 15)          # far-away pipes all look the same
        dy = max(-20, min(20, int(dy // self.dy_bucket)))  # clamp very high/low
        vel = int(vel // self.vel_bucket)
        return (dx, dy, vel)

    # ---------------- Choosing actions ----------------
    def choose_action(self, state, greedy=False):
        """
        Epsilon-greedy policy:
          with probability epsilon -> explore (random action)
          otherwise               -> exploit (action with the highest Q)
        greedy=True turns exploration off (used when showing off the agent).
        """
        if not greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        q_values = self.q[self.discretize(state)]
        if q_values[0] == q_values[1]:
            return 0  # tie (e.g. a brand-new state): default to not flapping
        return 0 if q_values[0] > q_values[1] else 1

    # ---------------- Learning ----------------
    def learn(self, state, action, reward, next_state, done):
        """
        The Q-learning update:
          Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
          target  = reward                          if the episode ended
                  = reward + gamma * max Q(s', a')  otherwise
        'target' is a better estimate than the old Q(s,a) because it includes
        the reward actually received.
        """
        s = self.discretize(state)
        s_next = self.discretize(next_state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q[s_next])
        self.q[s][action] += self.alpha * (target - self.q[s][action])

    def end_episode(self):
        """Explore a little less after every episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------------- Saving / loading ----------------
    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"q": dict(self.q), "epsilon": self.epsilon}, f)

    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q = defaultdict(lambda: [0.0] * N_ACTIONS, data["q"])
        self.epsilon = data["epsilon"]