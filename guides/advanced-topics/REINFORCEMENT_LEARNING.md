# Reinforcement Learning Guide

Master the art of training agents to make optimal decisions through trial and error.

## Table of Contents
1. [RL Fundamentals](#rl-fundamentals)
2. [Value-Based Methods](#value-based-methods)
3. [Policy Gradient Methods](#policy-gradient-methods)
4. [Actor-Critic Methods](#actor-critic-methods)
5. [Deep RL Algorithms](#deep-rl-algorithms)
6. [Advanced Topics](#advanced-topics)
7. [Practical RL](#practical-rl)

---

## RL Fundamentals

### Core Concepts

**Reinforcement Learning** = Learning by interacting with an environment to maximize rewards.

**Key Components:**
- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State (s)**: Current situation
- **Action (a)**: What the agent can do
- **Reward (r)**: Feedback signal
- **Policy (π)**: Agent's behavior (state → action mapping)
- **Value Function (V)**: Expected future rewards from a state
- **Q-Function (Q)**: Expected future rewards from state-action pair

**Mathematical Framework:**

```
Markov Decision Process (MDP): (S, A, P, R, γ)
- States: S (state space)
- Actions: A (action space)
- Transition: P(s'|s,a) - probability of reaching s' from s via action a
- Reward: R(s,a,s') - immediate reward for transition
- Discount: γ ∈ [0,1] - future reward discount factor

Markov Property: P(s_{t+1}|s_t, a_t, s_{t-1}, ..., s_0) = P(s_{t+1}|s_t, a_t)
(Future depends only on present, not past)

Goal: Find optimal policy π* that maximizes expected cumulative discounted reward:
  J(π) = E_π[Σ_{t=0}^∞ γ^t · r_t] = E_π[G_0]

where G_t = Σ_{k=0}^∞ γ^k · r_{t+k} is the return from time t

**Bellman Equations:**

Value Function V^π(s): Expected return starting from state s following policy π
  V^π(s) = E_π[G_t | s_t = s]
         = E_π[r_t + γ·V^π(s_{t+1}) | s_t = s]  (Bellman Expectation Equation)

Optimal Value Function:
  V*(s) = max_π V^π(s)
        = max_a E[r + γ·V*(s') | s, a]  (Bellman Optimality Equation)

Q-Function Q^π(s,a): Expected return starting from (s,a) following π
  Q^π(s,a) = E_π[G_t | s_t = s, a_t = a]
           = E[r + γ·V^π(s') | s, a]

Optimal Q-Function (Bellman Optimality for Q):
  Q*(s,a) = E[r + γ·max_a' Q*(s',a') | s, a]

Policy from Q-Function:
  π*(s) = argmax_a Q*(s,a)

**Key Relationships:**
- V^π(s) = Σ_a π(a|s) Q^π(s,a)
- Q^π(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γ·V^π(s')]
- Advantage Function: A^π(s,a) = Q^π(s,a) - V^π(s)
```

### Simple RL Environment

```python
import numpy as np
import gym

class SimpleGridWorld:
    """Custom grid world environment"""

    def __init__(self, size=5):
        self.size = size
        self.state = [0, 0]  # Start position
        self.goal = [size-1, size-1]  # Goal position

    def reset(self):
        self.state = [0, 0]
        return np.array(self.state)

    def step(self, action):
        """
        Actions: 0=up, 1=right, 2=down, 3=left
        """
        # Move
        if action == 0:  # up
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 1:  # right
            self.state[1] = min(self.size - 1, self.state[1] + 1)
        elif action == 2:  # down
            self.state[0] = min(self.size - 1, self.state[0] + 1)
        elif action == 3:  # left
            self.state[1] = max(0, self.state[1] - 1)

        # Reward
        if self.state == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # Small penalty for each step
            done = False

        return np.array(self.state), reward, done, {}

    def render(self):
        grid = np.zeros((self.size, self.size))
        grid[self.state[0], self.state[1]] = 1  # Agent
        grid[self.goal[0], self.goal[1]] = 2    # Goal
        print(grid)

# Usage
env = SimpleGridWorld(size=5)
state = env.reset()
done = False

while not done:
    action = np.random.randint(0, 4)  # Random action
    next_state, reward, done, _ = env.step(action)
    env.render()
```

---

## Value-Based Methods

### Q-Learning

**Core Idea:** Learn Q(s,a) - expected return from taking action a in state s.

**Q-Learning Update:**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

where:
- α: learning rate (step size)
- r: immediate reward
- γ: discount factor
- max_a' Q(s',a'): maximum Q-value at next state (bootstrap estimate)
- [r + γ max_a' Q(s',a') - Q(s,a)]: TD error

Temporal Difference (TD) Learning: Update Q toward TD target
  TD target = r + γ max_a' Q(s',a')
  TD error = TD target - Q(s,a)
```

**Convergence Guarantees:**
Q-Learning converges to Q* with probability 1 if:
1. All state-action pairs are visited infinitely often
2. Learning rate satisfies Robbins-Monro conditions:
   - Σ α_t = ∞ (learning never stops completely)
   - Σ α_t² < ∞ (learning rate decreases sufficiently)
   - Example: α_t = 1/t satisfies these conditions

**Key Properties:**
- Off-policy: Learns Q* regardless of behavior policy
- Model-free: No need for transition model P(s'|s,a)
- Bootstrapping: Uses estimate to update estimate
- Sample-based: Updates from single sample, not full expectation

```python
class QLearning:
    """Tabular Q-Learning"""

    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])  # Explore
        else:
            return np.argmax(self.Q[state])  # Exploit

    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        # Current Q-value
        current_q = self.Q[state, action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])

        # Update
        self.Q[state, action] += self.lr * (target_q - current_q)

    def train(self, env, episodes=1000):
        """Train the agent"""
        rewards_history = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Choose action
                action = self.get_action(state)

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Update Q-table
                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            rewards_history.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        return rewards_history

# Usage
env = gym.make('FrozenLake-v1')
agent = QLearning(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.1
)

rewards = agent.train(env, episodes=5000)
```

---

### Deep Q-Network (DQN)

**Key Innovation:** Use neural network to approximate Q-function for large state spaces.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with experience replay and target network"""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma

        # Networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def train_step(self, batch_size=64):
        """Single training step"""
        if len(self.memory) < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=1000, batch_size=64, update_target_every=10):
        """Train DQN agent"""
        rewards_history = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Select action
                action = self.select_action(state)

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Store transition
                self.memory.push(state, action, reward, next_state, float(done))

                # Train
                loss = self.train_step(batch_size)

                state = next_state
                total_reward += reward

            # Update target network
            if episode % update_target_every == 0:
                self.update_target_network()

            # Decay epsilon
            self.decay_epsilon()

            rewards_history.append(total_reward)

            if episode % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return rewards_history

# Usage
env = gym.make('CartPole-v1')
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=1e-3,
    gamma=0.99
)

rewards = agent.train(env, episodes=500)
```

---

### Double DQN

**Problem with DQN:** Overestimation of Q-values
**Solution:** Use policy network to select action, target network to evaluate

```python
class DoubleDQNAgent(DQNAgent):
    """Double DQN - reduces overestimation bias"""

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use policy network to select action, target network to evaluate
        with torch.no_grad():
            # Select best actions using policy network
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)

            # Evaluate selected actions using target network
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()

            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
```

---

## Policy Gradient Methods

### REINFORCE Algorithm

**Core Idea:** Directly optimize the policy using gradient ascent.

**Policy Gradient Theorem:**
```
∇J(θ) = E[∇log π(a|s,θ) * Q(s,a)]
```

```python
class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class REINFORCE:
    """REINFORCE algorithm"""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def train(self, env, episodes=1000):
        """Train REINFORCE agent"""
        rewards_history = []

        for episode in range(episodes):
            state = env.reset()
            log_probs = []
            rewards = []
            done = False

            # Collect trajectory
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state

            # Compute returns
            returns = self.compute_returns(rewards)

            # Policy gradient update
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)

            # Optimize
            self.optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            self.optimizer.step()

            total_reward = sum(rewards)
            rewards_history.append(total_reward)

            if episode % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        return rewards_history

# Usage
env = gym.make('CartPole-v1')
agent = REINFORCE(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=1e-3,
    gamma=0.99
)

rewards = agent.train(env, episodes=1000)
```

---

## Actor-Critic Methods

### Advantage Actor-Critic (A2C)

**Key Innovation:** Reduce variance by using a baseline (value function).

```python
class ActorCritic(nn.Module):
    """Actor-Critic network"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy) head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic (value) head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        action_probs = torch.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value

class A2C:
    """Advantage Actor-Critic"""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def train_step(self, state, action, reward, next_state, done, log_prob):
        """Single training step"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Get values
        _, value = self.model(state_tensor)
        _, next_value = self.model(next_state_tensor)

        # Compute advantage
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * next_value.item()

        advantage = td_target - value.item()

        # Actor loss (policy gradient with advantage)
        actor_loss = -log_prob * advantage

        # Critic loss (TD error)
        critic_loss = nn.MSELoss()(value, torch.tensor([[td_target]]))

        # Total loss
        loss = actor_loss + critic_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, env, episodes=1000):
        """Train A2C agent"""
        rewards_history = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Select action
                action, log_prob = self.select_action(state)

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Train
                loss = self.train_step(state, action, reward, next_state, done, log_prob)

                state = next_state
                total_reward += reward

            rewards_history.append(total_reward)

            if episode % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        return rewards_history
```

---

## Deep RL Algorithms

### Proximal Policy Optimization (PPO)

**Key Innovation:** Trust region optimization for stable policy updates.

```python
class PPO:
    """Proximal Policy Optimization"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 epsilon=0.2, c1=0.5, c2=0.01):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon  # Clipping parameter
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item()

    def compute_advantages(self, rewards, values, dones):
        """Compute GAE (Generalized Advantage Estimation)"""
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.gamma * values[i+1] - values[i]
                gae = delta + self.gamma * 0.95 * gae  # λ = 0.95

            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)

    def ppo_update(self, states, actions, old_log_probs, returns, advantages, epochs=10):
        """PPO update with multiple epochs"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(epochs):
            # Get current action probabilities and values
            action_probs, values = self.model(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()

            # Ratio for PPO clipping
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            critic_loss = nn.MSELoss()(values.squeeze(), returns)

            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

    def train(self, env, episodes=1000, steps_per_episode=1000):
        """Train PPO agent"""
        rewards_history = []

        for episode in range(episodes):
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

            state = env.reset()
            total_reward = 0

            for step in range(steps_per_episode):
                # Select action
                action, log_prob = self.select_action(state)

                # Get value
                with torch.no_grad():
                    _, value = self.model(torch.FloatTensor(state).unsqueeze(0))

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Store
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value.item())
                dones.append(done)

                state = next_state
                total_reward += reward

                if done:
                    break

            # Compute returns and advantages
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            advantages = self.compute_advantages(rewards, values, dones)

            # PPO update
            self.ppo_update(states, actions, log_probs, returns, advantages)

            rewards_history.append(total_reward)

            if episode % 20 == 0:
                avg_reward = np.mean(rewards_history[-20:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        return rewards_history

# Usage
env = gym.make('CartPole-v1')
agent = PPO(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=3e-4,
    gamma=0.99
)

rewards = agent.train(env, episodes=300)
```

---

### Deep Deterministic Policy Gradient (DDPG)

**For continuous action spaces**

```python
class DDPGActor(nn.Module):
    """DDPG Actor (deterministic policy)"""

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.network(state)

class DDPGCritic(nn.Module):
    """DDPG Critic (Q-function)"""

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class DDPG:
    """Deep Deterministic Policy Gradient"""

    def __init__(self, state_dim, action_dim, max_action, lr=1e-3, gamma=0.99, tau=0.005):
        # Actor
        self.actor = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = DDPGCritic(state_dim, action_dim)
        self.critic_target = DDPGCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

        # Replay buffer
        self.memory = ReplayBuffer(capacity=100000)

    def select_action(self, state, noise=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]

        # Add exploration noise
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

---

### Soft Actor-Critic (SAC)

**Key Innovation:** Maximum entropy RL for better exploration and stability.

**Objective:** Maximize expected return + entropy
```
J(π) = Σ_t E[(r_t + α·H(π(·|s_t)))]

where H(π(·|s)) = -Σ_a π(a|s) log π(a|s) is policy entropy
α: temperature parameter (controls exploration vs exploitation)
```

```python
class SACCritic(nn.Module):
    """Twin Q-networks for SAC"""

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 network (twin)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

class SACActor(nn.Module):
    """Gaussian policy for continuous actions"""

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        features = self.network(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stabilize training
        return mean, log_std

    def sample(self, state):
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()  # Sample with reparameterization

        # Tanh squashing for bounded actions
        action = torch.tanh(x) * self.max_action

        # Compute log probability (with correction for tanh squashing)
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

class SAC:
    """Soft Actor-Critic"""

    def __init__(self, state_dim, action_dim, max_action, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2):
        # Actor
        self.actor = SACActor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Twin Q-networks
        self.critic = SACCritic(state_dim, action_dim)
        self.critic_target = SACCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Hyperparameters
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Temperature parameter

        # Replay buffer
        self.memory = ReplayBuffer(capacity=1000000)

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if evaluate:
            # Deterministic action (mean) for evaluation
            with torch.no_grad():
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean) * self.max_action
        else:
            # Stochastic action for training
            with torch.no_grad():
                action, _ = self.actor.sample(state_tensor)

        return action.cpu().numpy()[0]

    def train_step(self, batch_size=256):
        if len(self.memory) < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q-values (minimum of twin Q-networks)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            # Add entropy term
            q_target = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * \
                      (q_next - self.alpha * next_log_probs)

        # Critic loss
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Maximize Q - α·log_prob (equivalent to minimizing negative)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

# Usage
env = gym.make('HalfCheetah-v2')  # Continuous control environment
agent = SAC(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    max_action=float(env.action_space.high[0]),
    lr=3e-4,
    gamma=0.99,
    alpha=0.2
)

# Training loop
for episode in range(1000):
    state = env.reset()
    episode_reward = 0

    for step in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.memory.push(state, action, reward, next_state, float(done))
        agent.train_step(batch_size=256)

        state = next_state
        episode_reward += reward

        if done:
            break

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
```

**SAC Advantages:**
- **Stable training** via twin Q-networks (reduces overestimation)
- **Better exploration** via entropy maximization
- **Off-policy** learning from replay buffer
- **Sample efficient** compared to on-policy methods
- **Automatic temperature tuning** possible (α becomes learnable parameter)

---

## Advanced Topics

### Multi-Agent RL

```python
class MultiAgentEnvironment:
    """Simple multi-agent environment"""

    def __init__(self, n_agents=2):
        self.n_agents = n_agents
        self.state_dim = 4
        self.action_dim = 2

    def reset(self):
        return [np.random.randn(self.state_dim) for _ in range(self.n_agents)]

    def step(self, actions):
        # Simulate multi-agent dynamics
        next_states = [np.random.randn(self.state_dim) for _ in range(self.n_agents)]
        rewards = [np.random.randn() for _ in range(self.n_agents)]
        done = False

        return next_states, rewards, done

class MADDPG:
    """Multi-Agent DDPG"""

    def __init__(self, n_agents, state_dim, action_dim):
        self.n_agents = n_agents
        self.agents = [
            DDPG(state_dim, action_dim, max_action=1.0)
            for _ in range(n_agents)
        ]

    def select_actions(self, states):
        return [agent.select_action(state) for agent, state in zip(self.agents, states)]

    def train_step(self, batch_size=64):
        for agent in self.agents:
            agent.train_step(batch_size)
```

---

### Hierarchical RL

```python
class HierarchicalAgent:
    """Hierarchical RL with options/skills"""

    def __init__(self, state_dim, n_options, option_dim):
        # High-level policy (selects options)
        self.meta_policy = PolicyNetwork(state_dim, n_options)

        # Low-level policies (execute options)
        self.option_policies = [
            PolicyNetwork(state_dim, option_dim)
            for _ in range(n_options)
        ]

    def select_action(self, state, option=None):
        if option is None:
            # Select option using meta-policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            option_probs = self.meta_policy(state_tensor)
            option = torch.multinomial(option_probs, 1).item()

        # Execute option
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.option_policies[option](state_tensor)
        action = torch.multinomial(action_probs, 1).item()

        return action, option
```

---

## Practical RL

### Custom Gym Environment

```python
import gym
from gym import spaces

class TradingEnv(gym.Env):
    """Custom trading environment"""

    def __init__(self, price_data):
        super().__init__()
        self.price_data = price_data
        self.current_step = 0

        # Action: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Observation: [price, position, cash]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.position = 0
        self.cash = 10000
        self.initial_cash = 10000

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.cash = self.initial_cash
        return self._get_observation()

    def step(self, action):
        current_price = self.price_data[self.current_step]

        # Execute action
        if action == 1 and self.cash >= current_price:  # Buy
            self.position += 1
            self.cash -= current_price
        elif action == 2 and self.position > 0:  # Sell
            self.position -= 1
            self.cash += current_price

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1

        # Reward: portfolio value change
        portfolio_value = self.cash + self.position * current_price
        reward = portfolio_value - self.initial_cash

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        current_price = self.price_data[self.current_step]
        return np.array([current_price, self.position, self.cash], dtype=np.float32)

# Usage
price_data = np.cumsum(np.random.randn(1000)) + 100
env = TradingEnv(price_data)

agent = PPO(state_dim=3, action_dim=3)
rewards = agent.train(env, episodes=500)
```

---

## Summary: Algorithm Selection

| Algorithm | Type | Action Space | Sample Efficiency | Stability |
|-----------|------|--------------|-------------------|-----------|
| **Q-Learning** | Value-based | Discrete | Low | High |
| **DQN** | Value-based | Discrete | Medium | Medium |
| **REINFORCE** | Policy gradient | Both | Low | Low |
| **A2C** | Actor-Critic | Both | Medium | Medium |
| **PPO** | Actor-Critic | Both | Medium | High |
| **DDPG** | Actor-Critic | Continuous | High | Medium |
| **SAC** | Actor-Critic | Continuous | High | High |
| **TD3** | Actor-Critic | Continuous | High | High |

---

## Key Takeaways

1. **Value-based** (DQN) for discrete actions, **Policy gradient** for continuous
2. **PPO** is the go-to algorithm for most applications (stable + effective)
3. **Experience replay** dramatically improves sample efficiency
4. **Target networks** stabilize training
5. **Exploration vs Exploitation** is critical - use epsilon-greedy or entropy bonus
6. **Normalize observations and rewards** for faster convergence
7. **Hyperparameters matter** - learning rate, discount factor, network architecture
8. **Start simple** - tabular methods → DQN → PPO

**Next Steps:**
- Implement PPO for a custom environment
- Try continuous control with DDPG or SAC
- Experiment with multi-agent RL
- Apply RL to real-world problems (robotics, trading, games)
- Read original papers (DQN, PPO, SAC)

---

## 📚 References

**Foundational Books:**

1. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   - The definitive textbook on RL. Available free at: http://incompleteideas.net/book/

2. **Bertsekas, D. P.** (2019). *Reinforcement Learning and Optimal Control*. Athena Scientific.
   - Rigorous mathematical treatment of RL

3. **Szepesvári, C.** (2010). *Algorithms for Reinforcement Learning*. Morgan & Claypool.
   - Concise introduction to RL algorithms

**Seminal Papers:**

**Value-Based Methods:**

1. **Watkins, C. J., & Dayan, P.** (1992). "Q-learning." *Machine Learning*, 8(3-4), 279-292.
   - Original Q-learning paper with convergence proof

2. **Mnih, V., Kavukcuoglu, K., Silver, D., et al.** (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
   - DQN playing Atari games

3. **Van Hasselt, H., Guez, A., & Silver, D.** (2016). "Deep reinforcement learning with double Q-learning." *AAAI 2016*.
   - Double DQN addressing overestimation bias

4. **Wang, Z., Schaul, T., Hessel, M., et al.** (2016). "Dueling network architectures for deep reinforcement learning." *ICML 2016*.
   - Dueling DQN architecture

**Policy Gradient Methods:**

5. **Williams, R. J.** (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256.
   - REINFORCE algorithm

6. **Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y.** (2000). "Policy gradient methods for reinforcement learning with function approximation." *NIPS 2000*.
   - Policy gradient theorem

7. **Schulman, J., Levine, S., Abbeel, P., et al.** (2015). "Trust region policy optimization." *ICML 2015*.
   - TRPO algorithm

8. **Schulman, J., Wolski, F., Dhariwal, P., et al.** (2017). "Proximal policy optimization algorithms." *arXiv:1707.06347*.
   - PPO algorithm (most widely used today)

**Actor-Critic Methods:**

9. **Mnih, V., Badia, A. P., Mirza, M., et al.** (2016). "Asynchronous methods for deep reinforcement learning." *ICML 2016*.
   - A3C algorithm

10. **Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al.** (2016). "Continuous control with deep reinforcement learning." *ICLR 2016*.
    - DDPG for continuous control

11. **Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S.** (2018). "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." *ICML 2018*.
    - SAC algorithm

12. **Fujimoto, S., van Hoof, H., & Meger, D.** (2018). "Addressing function approximation error in actor-critic methods." *ICML 2018*.
    - TD3 algorithm

**Advanced Topics:**

13. **Schaul, T., Quan, J., Antonoglou, I., & Silver, D.** (2016). "Prioritized experience replay." *ICLR 2016*.
    - Improved experience replay

14. **Hessel, M., Modayil, J., Van Hasselt, H., et al.** (2018). "Rainbow: Combining improvements in deep reinforcement learning." *AAAI 2018*.
    - Combining multiple DQN improvements

15. **Lowe, R., Wu, Y., Tamar, A., et al.** (2017). "Multi-agent actor-critic for mixed cooperative-competitive environments." *NIPS 2017*.
    - MADDPG for multi-agent settings

**Online Resources:**

- **OpenAI Spinning Up in Deep RL**: https://spinningup.openai.com/
  - Excellent educational resource with code implementations

- **DeepMind x UCL RL Lecture Series**: https://www.deepmind.com/learning-resources
  - Video lectures from leading researchers

- **CS285 at UC Berkeley**: Deep Reinforcement Learning (Sergey Levine)
  - Course materials and lectures

- **Gymnasium (formerly OpenAI Gym)**: https://gymnasium.farama.org/
  - Standard RL environments

- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
  - High-quality implementations of RL algorithms

**Key Conferences:**
- ICML, NeurIPS, ICLR (Machine Learning)
- AAAI, IJCAI (Artificial Intelligence)
- CoRL (Conference on Robot Learning)

---

*For more details on these algorithms, consult Sutton & Barto (2018) and the original papers listed above.*
