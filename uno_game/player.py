# uno_project/uno_game/player.py

# Import agent class for type hinting (optional but good practice)
# from .agent import UnoAgent # Causes circular import if agent imports Player - avoid for now

class Player:
    """Represents a player in the UNO game, potentially controlled by an RL agent."""
    def __init__(self, playerId, hand=None, nextPlayer=None, previousPlayer=None, agent=None):
        self.playerId = playerId
        self.hand = list(hand) if hand is not None else [] # Ensure mutable list copy
        self.nextPlayer = nextPlayer
        self.previousPlayer = previousPlayer
        self.agent = agent  # Can be None (human/scripted) or an UnoAgent instance
        self.last_state = None  # Store state before action
        self.last_action = None # Store action index taken

    def __str__(self):
        agent_info = f" (Agent {self.agent.agent_id} V{self.agent.variant})" if self.agent else ""
        return f"P{self.playerId}{agent_info}({len(self.hand)} cards)"

    def display_hand(self):
        return f"Player {self.playerId} Hand: {self.hand}"

def createPlayerList(count, agents=None):
    """Creates a circular doubly linked list of players, assigning agents if provided."""
    if count < 2:
        raise ValueError("Must have at least 2 players.")

    if agents and len(agents) != count:
        print(f"Warning: Number of agents ({len(agents)}) does not match player count ({count}). Agents will not be assigned correctly.")
        agents = None # Fallback to no agents

    head_agent = agents[0] if agents else None
    head = Player(0, [], None, None, head_agent)
    current = head
    for i in range(1, count):
        player_agent = agents[i] if agents else None
        # Link previous during creation
        newPlayer = Player(i, [], None, current, player_agent)
        current.nextPlayer = newPlayer
        current = newPlayer

    # Complete the circle
    head.previousPlayer = current
    current.nextPlayer = head
    print(f"Created {count} players. Player 0: {head}")
    return head