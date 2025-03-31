# uno_project/uno_game/game.py

import random as rand
# Use relative imports
from .cards import get_standard_deck, COLORS
from .player import Player
from .deck import Deck
# Import Agent class if needed for type checks (careful with circular imports)
# from .agent import UnoAgent

class ActiveGame:
    """Manages the state and logic of an UNO game, supporting RL agents."""

    def __init__(self, head_player, num_players, training_mode=True):
        self.deck = Deck(get_standard_deck())
        self.discard_pile = []
        self.current_player = head_player
        self.num_players = num_players
        self.active_players = num_players
        self.direction = 1
        self.draw_stack = 0
        self.draw_type = None
        self.chosen_color = None
        self.winners = [] # Player IDs in winning order
        self.turn = 0
        self.training_mode = training_mode # Enable/disable learning updates
        # Track rewards per step for agents if needed for analysis
        self.step_rewards = {} # {playerId: reward_this_step}

    # __str__ and other methods from previous version... keep them

    def start_game(self, initial_cards=7):
        """Initialize game: shuffle, deal cards, and set starting conditions."""
        print("Starting new UNO game...")
        self.deck = Deck(get_standard_deck()) # Reset deck
        self.discard_pile = []
        self.direction = 1
        self.draw_stack = 0
        self.draw_type = None
        self.chosen_color = None
        self.winners = []
        self.turn = 0
        self.active_players = self.num_players

        # Reset player hands and potentially last state/action if reusing player objects
        p = self.current_player
        for _ in range(self.num_players):
            p.hand = []
            p.last_state = None
            p.last_action = None
            if p.agent:
                 p.agent.total_rewards_history.append(0) # Reset game reward counter
            p = p.nextPlayer


        self.deck.shuffle()
        try:
            # Dealing modifies player hands directly
            self.deck.deal(self.current_player, initial_cards, self.num_players)
        except ValueError as e:
            print(f"Error starting game: {e}")
            return False

        # Flip the first card - handle special first cards
        try:
            self.flip_first_card()
        except RuntimeError as e:
            print(f"Error flipping first card: {e}")
            return False

        print("Game started!")
        # print(f"Initial state: {self}") # Can be verbose
        return True

    def flip_first_card(self):
        """Flips the first card onto the discard pile and handles its effect if special."""
        if self.deck.is_empty():
             raise RuntimeError("Cannot start game without a starting card.")

        first_card = self.deck.draw()
        while first_card and 'w+4' in first_card: # Rule: W+4 cannot be starting card
             print("First card was Wild Draw 4, reshuffling it in...")
             self.deck.add_cards([first_card])
             self.deck.shuffle()
             first_card = self.deck.draw()
             if self.deck.is_empty() and first_card and 'w+4' in first_card:
                 raise RuntimeError("Cannot start game, only Wild Draw 4 cards left?")
        
        if not first_card: # Should not happen after checks, but safeguard
             raise RuntimeError("Deck empty after trying to flip first card.")

        print(f"First card flipped: {first_card}")
        self.discard_pile.append(first_card)
        # Store initial chosen color if it's a wild
        if first_card[0] == 'w':
             # Let player 0 choose color (or use agent strategy)
             if self.current_player.agent:
                 self.chosen_color = self.current_player.agent.choose_color(self.current_player.hand)
             else:
                 self.chosen_color = random.choice(['r', 'b', 'g', 'y'])
             print(f"First card is Wild. Player {self.current_player.playerId} chooses {self.chosen_color}.")
        else:
             self.chosen_color = None # Ensure it's reset

        # --- Apply effect of first card ---
        # Note: The player who *would* play after the effect takes place starts.
        card_value = first_card[1:]

        if card_value == 's': # Skip
            print(f"First card is Skip. Player {self.current_player.playerId} is skipped.")
            self.current_player = self.current_player.nextPlayer # Next player starts
        elif card_value == 'rev': # Reverse
            print("First card is Reverse. Reversing direction.")
            self.direction *= -1
            if self.active_players == 2: # Reverse acts like Skip
                 print("Reverse with 2 players acts like Skip.")
                 self.current_player = self.current_player.nextPlayer
            # With >2 players, player 0 still starts, just goes the other way
        elif card_value == 'd2': # Draw Two
            # The first player must draw 2 and their turn is skipped.
            print(f"First card is Draw Two. Player {self.current_player.playerId} must draw 2 and is skipped.")
            # We don't call draw_card here to avoid agent step logic before game starts properly
            drawn = []
            for _ in range(2):
                 if self.deck.is_empty(): self.reshuffle()
                 if not self.deck.is_empty():
                     card = self.deck.draw()
                     self.current_player.hand.append(card)
                     drawn.append(card)
            print(f"Player {self.current_player.playerId} drew: {drawn}")
            self.current_player = self.current_player.nextPlayer # Next player starts
        elif first_card[0] == 'w' and '+4' not in first_card: # Normal Wild
            # Color was chosen above, player 0 starts
            pass
        # Number card: Player 0 starts normally.


    def play_turn(self):
        """Executes a single turn, using agent logic if available."""
        print("-" * 20)
        self.turn += 1
        current_p = self.current_player
        agent = current_p.agent
        print(f"Turn {self.turn} - Player {current_p.playerId}'s turn "
              f"{f'(Agent V{agent.variant})' if agent else ''}. Hand: {len(current_p.hand)}")
        # print(f"   Hand: {current_p.hand}") # Can be too verbose

        top_card = self.view_top_card()
        print(f"   Top card: {top_card}" + (f" (Chosen: {self.chosen_color})" if top_card and top_card.startswith('w') else ""))

        self.step_rewards = {p.playerId: 0 for p in self.get_all_players()} # Reset step rewards


        # --- Agent Decision Making ---
        state = None
        action_idx = -1 # Default action is "draw" or "no valid move"
        chosen_card = None

        if agent:
            state = agent.get_state(self, current_p)
            # Store state before action, used in agent.step later
            current_p.last_state = state

            # Check for forced draw first
            if self.draw_stack > 0:
                # Can the agent play a card to stack? (e.g., play D2 on D2)
                can_stack_cards = [c for c in current_p.hand if self.can_stack(c, self.draw_type)]
                if can_stack_cards:
                     print(f"   Player {current_p.playerId} can potentially stack.")
                     # Let agent decide whether to stack or draw (treat stacking as a normal move)
                     valid_moves = can_stack_cards
                     chosen_card, action_idx = agent.act(state, valid_moves, self) # Agent chooses best stack card
                     if chosen_card:
                          print(f"   Agent chose to stack with: {chosen_card}")
                          current_p.last_action = action_idx # Store the action index
                          self.play_card(chosen_card) # Play the stacking card
                     else:
                          # Agent decided not to stack (or act returned None) - force draw
                          print(f"   Agent chose not to stack or invalid action. Forcing draw.")
                          action_idx = -1 # Mark as draw action for learning
                          self.handle_forced_draw()
                else:
                    # Cannot stack, must draw
                    print(f"   Cannot stack {self.draw_type}. Must draw {self.draw_stack} cards.")
                    action_idx = -1 # Mark as draw action for learning
                    self.handle_forced_draw()

            else:
                # Normal turn, find valid moves
                valid_moves = [card for card in current_p.hand if self.is_valid_move(card, top_card)]
                print(f"   Valid moves: {valid_moves}")

                if valid_moves:
                    chosen_card, action_idx = agent.act(state, valid_moves, self)
                    if chosen_card:
                        print(f"   Agent chose card: {chosen_card} (Action Index: {action_idx})")
                        current_p.last_action = action_idx # Store action index for learning
                        self.play_card(chosen_card)
                    else:
                         # Agent chose not to play (e.g., epsilon-greedy chose invalidly, or network output weird)
                         # This should ideally not happen if act filters correctly, but handle it.
                         print(f"   Agent returned no valid card choice. Forcing draw.")
                         action_idx = -1
                         self.handle_no_valid_moves_draw()
                else:
                    # No valid moves, must draw
                    print(f"   No valid moves available.")
                    action_idx = -1
                    self.handle_no_valid_moves_draw()

        else:
            # --- Non-Agent Logic (e.g., random or simple heuristic) ---
            if self.draw_stack > 0:
                 can_stack_cards = [c for c in current_p.hand if self.can_stack(c, self.draw_type)]
                 if can_stack_cards:
                      chosen_card = random.choice(can_stack_cards) # Simple: always stack if possible
                      print(f"   Player {current_p.playerId} stacks with {chosen_card}")
                      self.play_card(chosen_card)
                 else:
                      print(f"   Player {current_p.playerId} must draw {self.draw_stack}")
                      self.handle_forced_draw()
            else:
                valid_moves = [card for card in current_p.hand if self.is_valid_move(card, top_card)]
                if valid_moves:
                    chosen_card = random.choice(valid_moves) # Simple random play
                    print(f"   Player {current_p.playerId} plays {chosen_card}")
                    self.play_card(chosen_card)
                else:
                    print(f"   Player {current_p.playerId} has no valid moves, drawing card.")
                    self.handle_no_valid_moves_draw()

        # --- Post-Turn Processing ---
        game_continues = self.check_win_and_game_over()

        # --- Update Agent Learning (if training and agent exists) ---
        # This happens *after* the action is completed and rewards are calculated
        if self.training_mode and agent and current_p.last_state is not None:
             reward = self.step_rewards.get(current_p.playerId, 0)
             next_state = agent.get_state(self, current_p) # Get state *after* the turn resolved
             done = not game_continues # Is the game over for this player? (Win/Loss handled separately too)

             # Special handling for win/loss rewards applied at end of game check
             # agent.step will handle the memory buffer and learning calls
             agent.step(current_p.last_state, current_p.last_action, reward, next_state, done)

             # Add reward to agent's total for this game
             if len(agent.total_rewards_history) > 0:
                 agent.total_rewards_history[-1] += reward


        return game_continues # Return True if game continues, False if over

    def handle_forced_draw(self):
        """Handles when a player *must* draw due to draw stack."""
        player = self.current_player
        draw_count = self.draw_stack
        print(f"   Player {player.playerId} draws {draw_count} cards due to {self.draw_type}.")
        self.draw_card(player, draw_count) # draw_card handles agent penalty internally
        self.draw_stack = 0
        self.draw_type = None
        self.chosen_color = None # Reset chosen color after penalty paid
        # Turn advances in draw_card

    def handle_no_valid_moves_draw(self):
        """Handles when a player has no moves and must draw ONE card."""
        player = self.current_player
        print(f"   Player {player.playerId} has no valid moves. Drawing one card.")
        drawn_card = self.draw_card(player, 1) # draw_card handles agent penalty

        if drawn_card:
            print(f"   Player {player.playerId} drew {drawn_card}")
            # Check if the drawn card can be played immediately
            top_card = self.view_top_card()
            if self.is_valid_move(drawn_card, top_card):
                print(f"   Player {player.playerId} can play the drawn card immediately!")
                # Need agent decision here too if agent exists!
                if player.agent:
                     # Should the agent *always* play the drawn card? Assume yes for now.
                     action_idx = player.agent._get_card_action_index(drawn_card)
                     player.last_action = action_idx # Update action for learning
                     # We need the state *before* this potential play for learning
                     # This complicates the flow - maybe don't learn on the immediate play?
                     # Or calculate reward for the combined draw+play action?
                     # Simple approach: Just play it, reward handled by play_card
                     self.play_card(drawn_card)
                else:
                     self.play_card(drawn_card) # Non-agent plays immediately
            else:
                print("   Drawn card cannot be played. Turn ends.")
                # Turn already advanced in draw_card if no immediate play
        else:
             print("   Deck is empty, cannot draw. Turn passes.")
             self.advance_turn() # Manually advance if draw failed


    def play_card(self, card):
        """Plays a card, updates state, calculates reward, applies effect, advances turn."""
        player = self.current_player
        agent = player.agent
        initial_hand_size = len(player.hand)

        if card not in player.hand:
             print(f"!!! Error: Player {player.playerId} tried to play {card} but doesn't have it. Hand: {player.hand}")
             # This can happen if agent's action mapping is wrong or state is stale.
             # To prevent crash, force draw? Or just skip turn?
             self.handle_no_valid_moves_draw() # Treat as if they had no valid move
             return

        player.hand.remove(card)
        self.discard_pile.append(card)
        print(f"   Played {card}. Hand size: {len(player.hand)}")

        # --- Calculate Reward for Playing Card ---
        reward = 0
        if self.training_mode and agent:
            reward += agent.reward_scales['play_card']
            # Reward for playing action cards?
            if any(c in card for c in ['s', 'rev', 'd2', '+4']):
                 reward += agent.reward_scales['play_action']
            if card[0] == 'w':
                 reward += agent.reward_scales['play_wild']
            # Reward for reducing hand size (more reward closer to 0)
            reward += agent.reward_scales['reduce_hand'] * (initial_hand_size - len(player.hand)) / initial_hand_size

            self.step_rewards[player.playerId] = self.step_rewards.get(player.playerId, 0) + reward


        # --- Apply Card Effects ---
        card_color = card[0]
        card_value = card[1:]
        self.chosen_color = None # Reset chosen color by default
        next_player_affected = True # Flag if the *next* player action is determined here

        if card_value == 's':
            self.apply_skip()
        elif card_value == 'rev':
            self.apply_reverse()
        elif card_value == 'd2':
            self.apply_draw_two(card)
        elif card_color == 'w':
            # Agent chooses color
            if agent:
                self.chosen_color = agent.choose_color(player.hand)
            else: # Random choice for non-agents
                self.chosen_color = random.choice(['r', 'b', 'g', 'y'])
            print(f"   Player {player.playerId} chose color: {self.chosen_color}")

            if card_value == '+4':
                self.apply_wild_draw_four(card)
            else: # Regular wild, turn just advances
                self.advance_turn()
                next_player_affected = False
        else: # Number card
            self.advance_turn()
            next_player_affected = False

        # Update learning *after* effects applied and turn potentially advanced
        # (Moved step update to end of play_turn)


    def draw_card(self, player, count):
        """Draws cards, handles reshuffle, calculates penalty. Returns last drawn card or None."""
        drawn_cards_list = []
        for i in range(count):
            if self.deck.is_empty():
                if not self.reshuffle():
                    print(f"   Cannot draw card {i+1}/{count}. No cards left.")
                    break # Stop drawing if reshuffle failed
            # Draw after potential reshuffle
            if not self.deck.is_empty():
                 drawn_card = self.deck.draw()
                 player.hand.append(drawn_card)
                 drawn_cards_list.append(drawn_card)
            else:
                 print(f"   Cannot draw card {i+1}/{count}. Deck empty even after reshuffle attempt.")
                 break

        # --- Calculate Penalty for Drawing ---
        if self.training_mode and player.agent:
             penalty = player.agent.reward_scales['draw_card'] * len(drawn_cards_list)
             self.step_rewards[player.playerId] = self.step_rewards.get(player.playerId, 0) + penalty
             print(f"   Applied draw penalty: {penalty:.2f}")


        # Do NOT advance turn here automatically if it's a draw during the player's turn action phase
        # Turn advancement should be handled by the calling context (handle_forced_draw, handle_no_valid_moves_draw)
        # However, if called directly (e.g. initial D2), maybe it should advance? Needs context.
        # Let's assume draw_card itself doesn't advance the turn.

        return drawn_cards_list[-1] if drawn_cards_list else None


    def is_valid_move(self, card, top_card):
        """Checks if a card is playable on the top card."""
        if not top_card: return True # Should only happen if discard is empty (e.g., start?)

        card_color = card[0]
        card_value = card[1:]

        # Handle Wild on top
        if top_card.startswith('w'):
            # Must match chosen color OR be another Wild
            return card_color == self.chosen_color or card_color == 'w'

        # Standard play
        top_color = top_card[0]
        top_value = top_card[1:]

        # Match color OR match value/type OR play a Wild
        return card_color == top_color or card_value == top_value or card_color == 'w'

    def can_stack(self, card, draw_type):
        """Check if 'card' can be played to stack on 'draw_type'."""
        if draw_type == 'd2' and 'd2' in card:
            return True
        if draw_type == 'w+4' and 'w+4' in card:
            return True
        # Add house rules here, e.g., D2 on W+4?
        return False

    # --- Effect Application Methods (apply_skip, apply_reverse, etc.) ---
    # These methods should primarily update game state (direction, draw_stack)
    # and call advance_turn().
    def advance_turn(self):
        if self.active_players <= 0: return
        if self.direction == 1:
            self.current_player = self.current_player.nextPlayer
        else:
            self.current_player = self.current_player.previousPlayer

    def apply_skip(self):
        print(f"   Skipping next player...")
        self.advance_turn() # Move to skipped player
        print(f"   Player {self.current_player.playerId} is skipped.")
        self.advance_turn() # Move to the player *after* the skipped one

    def apply_reverse(self):
        print("   Reversing direction!")
        self.direction *= -1
        if self.active_players == 2:
             print("   Reverse acts like Skip with 2 players.")
             self.advance_turn() # Move to the other player
             print(f"   Player {self.current_player.playerId} is skipped.")
             self.advance_turn() # Move back to original player (effectively skipping)
        else:
             self.advance_turn() # Move to the next player in the new direction

    def apply_draw_two(self, card):
        print("   Draw Two played!")
        if self.draw_stack > 0 and self.draw_type == 'd2':
            self.draw_stack += 2
            print(f"   Stacking! Draw stack is now {self.draw_stack}")
        # elif add house rules for stacking on W+4 if desired
        else:
            if self.draw_stack > 0: print(f"   Cannot stack {card} on {self.draw_type}. Resetting.")
            self.draw_stack = 2
            self.draw_type = 'd2'
        self.advance_turn() # Move to the player who needs to draw or stack

    def apply_wild_draw_four(self, card):
         print("   Wild Draw Four played!")
         if self.draw_stack > 0 and self.draw_type == 'w+4':
             self.draw_stack += 4
             print(f"   Stacking! Draw stack is now {self.draw_stack}")
         # elif add house rules for stacking on D2 if desired
         else:
             if self.draw_stack > 0: print(f"   Cannot stack {card} on {self.draw_type}. Resetting.")
             self.draw_stack = 4
             self.draw_type = 'w+4'
         self.advance_turn() # Move to the player who needs to draw or stack


    def reshuffle(self):
        """Reshuffles discard pile into the deck."""
        print("--- Reshuffling ---")
        if len(self.discard_pile) <= 1:
            print("   Not enough cards in discard pile to reshuffle.")
            return False

        top_card = self.discard_pile.pop()
        cards_to_shuffle = self.discard_pile
        self.discard_pile = [top_card] # Reset discard pile

        print(f"   Adding {len(cards_to_shuffle)} cards back to deck.")
        self.deck.add_cards(cards_to_shuffle)
        self.deck.shuffle()
        print("--- Reshuffle Complete ---")
        return True

    def view_top_card(self):
        return self.discard_pile[-1] if self.discard_pile else None

    def remove_player(self, player_to_remove):
        """Removes a player from the game loop (after they win)."""
        if self.active_players <= 1: return

        print(f"--- Player {player_to_remove.playerId} has finished! Removing from game. ---")
        prev_p = player_to_remove.previousPlayer
        next_p = player_to_remove.nextPlayer

        prev_p.nextPlayer = next_p
        next_p.previousPlayer = prev_p

        # If the removed player *was* the current player (which happens right after their winning move)
        # advance turn needs to happen carefully based on direction AFTER removal.
        # We'll let the main loop handle advancing from the 'new' current player setup by the links.

        self.active_players -= 1
        print(f"   {self.active_players} players remaining.")

        # Important: If the player being removed IS self.current_player,
        # self.current_player needs to be updated *here* to the next logical player.
        if self.current_player == player_to_remove:
             self.current_player = next_p # Assume normal flow continues


    def check_win_and_game_over(self):
         """Checks if current player won and if the game is over. Returns True if game continues."""
         player = self.current_player # Player who just acted or drew
         game_over = False

         # Check if the player just played their last card
         if not player.hand:
             print(f"!!! Player {player.playerId} played their last card!")
             self.winners.append(player.playerId)
             if player.agent:
                 player.agent.record_win()
                 # Apply WIN reward (handled at end of episode now)
                 # self.step_rewards[player.playerId] = self.step_rewards.get(player.playerId, 0) + player.agent.reward_scales['win']


             # Remove player from rotation
             self.remove_player(player)

             # Check if game ends (only 1 player left)
             if self.active_players <= 1:
                 game_over = True
                 if self.active_players == 1:
                      # Last remaining player is the loser
                      loser = self.current_player # The only one left
                      print(f"!!! Player {loser.playerId} is the last remaining player (loser).")
                      if loser.playerId not in self.winners:
                           self.winners.append(loser.playerId)
                      # Apply LOSE reward (handled at end of episode now)
                      # if self.training_mode and loser.agent:
                      #     self.step_rewards[loser.playerId] = self.step_rewards.get(loser.playerId, 0) + loser.agent.reward_scales['lose']


         if game_over:
             print("="*30)
             print("GAME OVER!")
             print(f"Final Ranking (First to Last): {self.winners}")
             print("="*30)

             # --- Apply final win/loss rewards ---
             if self.training_mode:
                 all_players = self.get_all_players(include_winners=True) # Need original list
                 winner_id = self.winners[0]
                 loser_id = self.winners[-1] if len(self.winners) == self.num_players else None

                 for p in all_players:
                      if p.agent and p.last_state is not None:
                           final_reward = 0
                           is_done = True # Game is done for everyone now
                           if p.playerId == winner_id:
                                final_reward = p.agent.reward_scales['win']
                                print(f"Applying final WIN reward {final_reward} to P{p.playerId}")
                           elif p.playerId == loser_id:
                                final_reward = p.agent.reward_scales['lose']
                                print(f"Applying final LOSE reward {final_reward} to P{p.playerId}")
                           # Optionally add rewards/penalties for 2nd, 3rd place etc.

                           next_state = p.agent.get_state(self, p) # Get final state
                           p.agent.step(p.last_state, p.last_action, final_reward, next_state, is_done)
                           if len(p.agent.total_rewards_history) > 0:
                                p.agent.total_rewards_history[-1] += final_reward


             return False # Game does not continue

         return True # Game continues


    def get_all_players(self, include_winners=False):
        """ Utility to get a list of all player objects currently in the game """
        # This needs modification if players are truly removed.
        # Assuming linked list remains intact but self.active_players decreases.
        # If remove_player breaks links permanently, need another way to track all original players.
        # Let's assume we start at current and loop N times for original players if include_winners=True
        players = []
        start_node = self.current_player
        p = start_node
        if include_winners:
             # Try to iterate through the original number of players, hoping links are still valid enough
             # This is fragile if remove_player modifies links drastically.
             # A better way would be to store the initial list of players.
             for _ in range(self.num_players):
                  players.append(p)
                  p = p.nextPlayer
                  if p == start_node and len(players) < self.num_players:
                       print("Warning: Player loop shorter than expected in get_all_players(include_winners=True)")
                       break # Avoid infinite loop if structure is broken
        else:
             # Get only active players
             for _ in range(self.active_players):
                  players.append(p)
                  p = p.nextPlayer
                  if p == start_node and len(players) < self.active_players:
                       print("Warning: Player loop shorter than expected in get_all_players(active)")
                       break
        return players


    def run_game_loop(self, max_turns=500):
        """Runs the main game loop until the game ends or max turns reached."""
        if not self.start_game():
            print("Failed to initialize game.")
            return False # Game did not start

        game_continues = True
        while game_continues:
            game_continues = self.play_turn()
            if self.turn >= max_turns:
                print(f"\nReached maximum turns ({max_turns}). Game ended forcibly.")
                # Assign final rewards based on hand size? Penalize everyone?
                if self.training_mode:
                    all_players = self.get_all_players(include_winners=True)
                    for p in all_players:
                        if p.agent and p.last_state is not None:
                             # Apply a penalty for not finishing
                             final_reward = p.agent.reward_scales['lose'] / 2 # Half losing penalty
                             next_state = p.agent.get_state(self, p)
                             p.agent.step(p.last_state, p.last_action, final_reward, next_state, True)
                             if len(p.agent.total_rewards_history) > 0:
                                  p.agent.total_rewards_history[-1] += final_reward


                game_continues = False # End the loop

        # Game over message is printed by check_win_and_game_over
        # Return True if game completed (even by max turns), False if setup failed?
        return True