# uno_project/uno_game/agent.py

from time import sleep
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np
import copy

# Use relative import for card info if needed, though encode_card is self-contained now
# from .cards import get_standard_deck # Example if needed elsewhere

# --- Constants ---
STATE_SIZE = 89  # Calculated based on get_state implementation
ACTION_SIZE = 54 # Based on the hand encoding indices (13 types * 4 colors + 2 wild types)
# Note: This means the agent selects a *type* of card, and the game logic maps it back if available/valid.


# --- Experience Replay ---
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# --- Card Encoding ---
def encode_card(card):
    """Convert card string ('r1', 'w', 'b+2') to a fixed-size vector."""
    # Simplified encoding for state - just presence/absence might be enough
    # Color encoding: r, b, g, y, w (5 bits)
    color_encoding = [0] * 5
    if not card: # Handle potentially empty top card at start?
        return color_encoding + [0] * 14 # Return zero vector

    color = card[0]
    if color == 'r': color_encoding[0] = 1
    elif color == 'b': color_encoding[1] = 1
    elif color == 'g': color_encoding[2] = 1
    elif color == 'y': color_encoding[3] = 1
    elif color == 'w': color_encoding[4] = 1

    # Type encoding: 0-9, s, rev, d2, w, w+4 (14 bits)
    type_encoding = [0] * 14
    value = card[1:]

    if color == 'w':
        if value == '+4': type_encoding[13] = 1 # Wild Draw 4
        else: type_encoding[12] = 1 # Wild
    else:
        if value == 's': type_encoding[10] = 1 # Skip
        elif value == 'rev': type_encoding[11] = 1 # Reverse
        elif value == 'd2': type_encoding[12] = 1 # Draw 2 (Treating same index as Wild for simplicity here - adjust if needed)
        else:
            try:
                num = int(value)
                if 0 <= num <= 9:
                    type_encoding[num] = 1
            except ValueError:
                pass # Should not happen with valid cards

    return color_encoding + type_encoding

# --- Q-Network ---
class UnoQNetwork(nn.Module):
    """Neural Network model for approximating Q-values."""
    def __init__(self, input_size, output_size, hidden_size=128):
        super(UnoQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- RL Agent ---
class UnoAgent:
    """Interacts with and learns from the UNO environment."""
    def __init__(self, agent_id, variant=0, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        # Hyperparameters based on variant
        variant_params = [
            {'lr': 0.001, 'gamma': 0.99, 'epsilon': 1.0, 'hidden': 128, 'mem_cap': 10000, 'batch': 64},  # Balanced
            {'lr': 0.002, 'gamma': 0.98, 'epsilon': 1.0, 'hidden': 128, 'mem_cap': 10000, 'batch': 64},  # Higher LR
            {'lr': 0.0005,'gamma': 0.99, 'epsilon': 1.0, 'hidden': 128, 'mem_cap': 10000, 'batch': 64},  # Lower LR
            {'lr': 0.001, 'gamma': 0.95, 'epsilon': 1.0, 'hidden': 128, 'mem_cap': 10000, 'batch': 64},  # Lower Gamma
            {'lr': 0.001, 'gamma': 0.99, 'epsilon': 0.9, 'hidden': 128, 'mem_cap': 10000, 'batch': 64},  # Lower Epsilon start
            {'lr': 0.001, 'gamma': 0.99, 'epsilon': 1.0, 'hidden': 64,  'mem_cap': 10000, 'batch': 64},  # Smaller Net
            {'lr': 0.001, 'gamma': 0.99, 'epsilon': 1.0, 'hidden': 256, 'mem_cap': 10000, 'batch': 64},  # Larger Net
            {'lr': 0.001, 'gamma': 0.99, 'epsilon': 1.0, 'hidden': 128, 'mem_cap': 20000, 'batch': 128}, # Larger Memory/Batch
            {'lr': 0.001, 'gamma': 0.99, 'epsilon': 1.0, 'hidden': 128, 'mem_cap': 10000, 'batch': 64}, # Focus: Win Reward
            {'lr': 0.001, 'gamma': 0.99, 'epsilon': 1.0, 'hidden': 128, 'mem_cap': 10000, 'batch': 64}  # Focus: Hand Size Reward
        ]

        params = variant_params[variant % len(variant_params)] # Cycle through variants if agent_id > 9

        self.agent_id = agent_id
        self.variant = variant
        self.state_size = state_size
        self.action_size = action_size # Number of distinct card types/actions agent can choose
        self.gamma = params['gamma']        # Discount factor
        self.epsilon = params['epsilon']    # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 # Slower decay for more games
        self.learning_rate = params['lr']
        self.memory_capacity = params['mem_cap']
        self.batch_size = params['batch']
        self.hidden_size = params['hidden']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent {self.agent_id} (Variant {self.variant}) using device: {self.device}")
        sleep(2)

        # Reward structure
        self.reward_scales = {
            'play_card': 0.1,       # Small reward for playing any card
            'play_action': 0.2,     # Slightly higher for action cards
            'play_wild': 0.15,       # Reward for playing wild
            'reduce_hand': 0.5,     # Reward scaled by 1/(hand_size+1)
            'draw_card': -0.2,      # Penalty per card drawn
            'win': 20.0,            # Large reward for winning
            'lose': -10.0           # Large penalty for losing (being last)
        }
        # Variant specific reward tweaks
        if variant == 8: # Focus Win
            self.reward_scales['win'] = 30.0
            self.reward_scales['lose'] = -15.0
        elif variant == 9: # Focus Hand Size
             self.reward_scales['reduce_hand'] = 1.0
             self.reward_scales['play_card'] = 0.05 # De-emphasize just playing

        # Q-Network and Target Network
        self.qnetwork = UnoQNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_network = UnoQNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.qnetwork.state_dict())
        self.target_network.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.memory_capacity)
        self.update_every = 4  # How often to update the network
        self.tau = 1e-3        # For soft update of target parameters
        self.steps = 0         # Counter for learning steps

        # Tracking metrics
        self.games_played = 0
        self.games_won = 0
        self.total_rewards_history = [] # Track rewards per game


    def _get_card_action_index(self, card):
        """Maps a card string to its corresponding action index (0-53)."""
        # This mapping needs to be consistent with hand_encoding in get_state
        # 0-12: Red (0-9, S, R, D2)
        # 13-25: Blue
        # 26-38: Green
        # 39-51: Yellow
        # 52: Wild
        # 53: Wild Draw 4
        color = card[0]
        value = card[1:]
        offset = 0
        type_idx = -1

        if color == 'r': offset = 0
        elif color == 'b': offset = 13
        elif color == 'g': offset = 26
        elif color == 'y': offset = 39
        elif color == 'w':
            return 53 if value == '+4' else 52
        else:
            return -1 # Invalid color

        if value == 's': type_idx = 10
        elif value == 'rev': type_idx = 11
        elif value == 'd2': type_idx = 12
        else:
            try:
                type_idx = int(value)
            except ValueError:
                return -1 # Invalid value

        if 0 <= type_idx <= 12:
            return offset + type_idx
        else:
            return -1 # Invalid type index

    def get_state(self, game, player):
        """Converts the current game state into a tensor for the agent."""
        state = []

        # 1. Top card encoding (19 bits)
        top_card = game.view_top_card()
        state.extend(encode_card(top_card))

        # 2. Current chosen color (4 bits: r, b, g, y)
        color_encoding = [0, 0, 0, 0]
        current_effective_color = game.chosen_color if game.chosen_color else (top_card[0] if top_card and top_card[0] != 'w' else None)
        if current_effective_color == 'r': color_encoding[0] = 1
        elif current_effective_color == 'b': color_encoding[1] = 1
        elif current_effective_color == 'g': color_encoding[2] = 1
        elif current_effective_color == 'y': color_encoding[3] = 1
        # If no color (start, or wild just played before choice?), all zeros might be okay
        state.extend(color_encoding)

        # 3. Player's hand encoding (54 bits - presence of each card type)
        hand_encoding = [0] * ACTION_SIZE # Size 54
        for card in player.hand:
            idx = self._get_card_action_index(card)
            if idx != -1:
                hand_encoding[idx] = 1 # Mark presence
        state.extend(hand_encoding)

        # 4. Game state features (3 bits)
        state.append(game.direction)  # Play direction (+1 or -1)
        state.append(game.draw_stack / 10.0) # Normalize draw stack (e.g., max stack 10?)
        state.append(len(player.hand) / 20.0) # Normalize hand size (e.g., max 20 cards?)

        # 5. Other players' hand sizes (9 bits, normalized)
        other_hand_sizes = []
        other_player = player.nextPlayer
        count = 0
        max_opponents = 9 # Max 10 players total - 1 self
        while other_player != player and count < max_opponents:
             # Check if other player is still active (has a hand or is in the game loop)
             # This requires modification in ActiveGame.remove_player or how players are tracked.
             # Assuming for now all players in the list are active until they win.
            other_hand_sizes.append(len(other_player.hand) / 20.0) # Normalize
            other_player = other_player.nextPlayer
            count += 1
        # Pad if fewer than max opponents
        while len(other_hand_sizes) < max_opponents:
            other_hand_sizes.append(0.0)
        state.extend(other_hand_sizes)

        # Ensure state size matches STATE_SIZE
        if len(state) != STATE_SIZE:
             print(f"Warning: State size mismatch! Expected {STATE_SIZE}, got {len(state)}")
             # Pad or truncate if necessary, though ideally get_state is correct
             state = state[:STATE_SIZE] + [0.0] * (STATE_SIZE - len(state))


        return torch.FloatTensor(state).unsqueeze(0).to(self.device)


    def act(self, state, valid_cards, game_instance):
        """Returns action index for the given state as per current policy."""
        # Map valid cards to their action indices
        valid_action_indices = [idx for card in valid_cards if (idx := self._get_card_action_index(card)) != -1]

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Choose randomly from the *valid* actions/cards
             if not valid_action_indices:
                 chosen_action_index = -1 # Indicate no valid action (draw)
             else:
                 chosen_action_index = random.choice(valid_action_indices)
        else:
            # Choose the best action from the Q-network, considering only valid actions
            self.qnetwork.eval() # Set network to evaluation mode
            with torch.no_grad():
                action_values = self.qnetwork(state) # Get Q-values for all 54 actions
            self.qnetwork.train() # Set network back to train mode

            # Filter Q-values to only include valid actions
            valid_q_values = {idx: action_values[0][idx].item() for idx in valid_action_indices}

            if not valid_q_values:
                 chosen_action_index = -1 # Indicate no valid action (draw)
            else:
                # Choose the action index with the highest Q-value among valid actions
                 chosen_action_index = max(valid_q_values, key=valid_q_values.get)

        # Map chosen action index back to a specific card if possible (needed?)
        # The game logic needs the *card string*. We find a card in hand matching the chosen index.
        chosen_card = None
        if chosen_action_index != -1:
             for card in valid_cards:
                  if self._get_card_action_index(card) == chosen_action_index:
                       chosen_card = card
                       break # Take the first match

        # It's possible chosen_card is None if mapping failed or index was -1
        # Return both the chosen card string (for game logic) and the action index (for learning)
        return chosen_card, chosen_action_index


    def choose_color(self, player_hand):
        """Agent chooses a color when playing a Wild card."""
        # Simple strategy: choose the color the agent has the most of (excluding wilds)
        color_counts = {'r': 0, 'b': 0, 'g': 0, 'y': 0}
        for card in player_hand:
            if card[0] in color_counts:
                color_counts[card[0]] += 1

        if not any(color_counts.values()): # If only wilds left
            return random.choice(['r', 'b', 'g', 'y'])
        else:
            return max(color_counts, key=color_counts.get)


    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience
        if action is not None and action != -1: # Only store experiences where an action was taken
            self.memory.push(state, action, reward, next_state, done)

            self.steps += 1
            # Learn every UPDATE_EVERY time steps.
            if self.steps % self.update_every == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample(self.batch_size)
                    self.learn(experiences)

        # Decay epsilon after each game completion (or periodically)
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.games_played += 1


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""
        states = torch.cat([e.state for e in experiences if e is not None]).to(self.device)
        actions = torch.tensor([e.action for e in experiences if e is not None], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences if e is not None], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.cat([e.next_state for e in experiences if e is not None]).to(self.device)
        dones = torch.tensor([e.done for e in experiences if e is not None], dtype=torch.float).unsqueeze(1).to(self.device)

        # Get max predicted Q values for next states from target model
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional, can help stability)
        # torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), 1.0)
        self.optimizer.step()

        # --- Soft update target network --- #
        self.soft_update()

    def soft_update(self):
        """Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(), self.qnetwork.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def record_win(self):
        self.games_won += 1

    def save(self, filename):
        """Save the agent's Q-network weights."""
        torch.save(self.qnetwork.state_dict(), filename)
        print(f"Agent {self.agent_id} (Variant {self.variant}) model saved to {filename}")

    def load(self, filename):
        """Load Q-network weights from a file."""
        try:
             self.qnetwork.load_state_dict(torch.load(filename, map_location=self.device))
             self.target_network.load_state_dict(self.qnetwork.state_dict()) # Keep target sync'd
             self.qnetwork.eval() # Set to eval mode after loading if not training
             self.target_network.eval()
             print(f"Agent {self.agent_id} (Variant {self.variant}) model loaded from {filename}")
        except FileNotFoundError:
             print(f"Error: Could not find model file {filename} for agent {self.agent_id}")
        except Exception as e:
             print(f"Error loading model for agent {self.agent_id}: {e}")