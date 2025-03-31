# uno_game/deck.py

import random as rand
# Use relative import since cards.py is in the same package
from .cards import get_standard_deck

class Deck:
    """Represents the UNO draw pile."""
    def __init__(self, cards=None):
        # If no specific cards are passed, use a standard deck
        self.cards = cards if cards is not None else get_standard_deck()

    def shuffle(self):
        """Shuffles the deck."""
        rand.shuffle(self.cards)
        print("Deck shuffled.")

    def deal(self, start_player, num_cards_per_player, num_players):
        """Deals cards to players."""
        if len(self.cards) < num_cards_per_player * num_players:
            raise ValueError("Not enough cards in the deck to deal.")

        current_player = start_player
        for _ in range(num_cards_per_player):
            for _ in range(num_players):
                if not self.cards: # Should not happen with initial check, but safe
                     print("Warning: Deck ran out during initial deal.")
                     return start_player # Return potentially partially dealt players
                current_player.hand.append(self.cards.pop())
                current_player = current_player.nextPlayer
        return start_player # Return the starting player head

    def draw(self):
        """Draws a single card from the deck. Returns None if empty."""
        if not self.cards:
            return None
        return self.cards.pop()

    def is_empty(self):
        """Checks if the deck is empty."""
        return len(self.cards) == 0

    def add_cards(self, cards_to_add):
        """Adds a list of cards to the deck (e.g., from discard pile)."""
        self.cards.extend(cards_to_add)

    def __len__(self):
        return len(self.cards)

# Example usage (optional, for testing this module)
if __name__ == '__main__':
    from .player import createPlayerList # Need this for dealing example
    my_deck = Deck()
    print(f"Initial deck size: {len(my_deck)}")
    my_deck.shuffle()
    players = createPlayerList(4)
    my_deck.deal(players, 7, 4)
    print(f"Deck size after dealing: {len(my_deck)}")
    print(f"Player 0 hand size: {len(players.hand)}")
    print(f"Player 1 hand size: {len(players.nextPlayer.hand)}")