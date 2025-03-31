# uno_project/uno_game/cards.py

# Define constants for colors and special card types for clarity and potential reuse.
# Although not strictly used within this file's core function,
# it's a logical place to keep card-related constants.
COLORS = {'red': 'r', 'blue': 'b', 'green': 'g', 'yellow': 'y', 'wild': 'w'}
CARD_TYPES = {
    'skip': 's',
    'reverse': 'rev',
    'draw_two': 'd2',
    'wild': 'w',
    'wild_draw_four': 'w+4'
    # Numbers 0-9 are represented directly
}

def get_standard_deck():
    """
    Generates a standard 108-card UNO deck as a list of strings.

    Returns:
        list: A list containing 108 card strings representing a standard UNO deck.
              Example card strings: 'r1', 'g0', 'bs', 'yrev', 'bd2', 'w', 'w+4'.
    """
    cards = []
    colors_for_numbers_actions = ['r', 'b', 'g', 'y'] # Exclude wild for these

    # --- Number Cards (76 cards) ---
    for color in colors_for_numbers_actions:
        # One '0' card per color
        cards.append(f"{color}0")
        # Two of each '1' through '9' per color
        for num in range(1, 10):
            cards.append(f"{color}{num}")
            cards.append(f"{color}{num}")

    # --- Action Cards (24 cards) ---
    for color in colors_for_numbers_actions:
        # Two Skip cards per color
        cards.extend([f"{color}{CARD_TYPES['skip']}", f"{color}{CARD_TYPES['skip']}"])
        # Two Reverse cards per color
        cards.extend([f"{color}{CARD_TYPES['reverse']}", f"{color}{CARD_TYPES['reverse']}"])
        # Two Draw Two cards per color
        cards.extend([f"{color}{CARD_TYPES['draw_two']}", f"{color}{CARD_TYPES['draw_two']}"])

    # --- Wild Cards (8 cards) ---
    # Four Wild cards
    cards.extend([CARD_TYPES['wild']] * 4)
    # Four Wild Draw Four cards
    cards.extend([CARD_TYPES['wild_draw_four']] * 4)

    # The list `cards` should now contain 108 card strings.
    return cards

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    # This block runs only when cards.py is executed directly (e.g., python uno_game/cards.py)
    standard_deck = get_standard_deck()
    deck_size = len(standard_deck)

    print(f"Generated a standard UNO deck with {deck_size} cards.")

    # Optional: Print card counts for verification
    if deck_size == 108:
        print("Deck size is correct (108 cards).")
        # Count specific card types for a quick check
        zero_cards = sum(1 for card in standard_deck if card.endswith('0'))
        red_cards = sum(1 for card in standard_deck if card.startswith('r'))
        wild_draw_four_cards = sum(1 for card in standard_deck if card == CARD_TYPES['wild_draw_four'])

        print(f" - Found {zero_cards} '0' cards (Expected: 4)")
        print(f" - Found {red_cards} Red cards (Expected: 19 numbers + 2*3 actions = 25)")
        print(f" - Found {wild_draw_four_cards} Wild Draw Four cards (Expected: 4)")
    else:
        print(f"Error: Expected deck size 108, but got {deck_size}.")

    # Optional: Print the first few cards
    # print("\nFirst 10 cards in generated deck (unshuffled):")
    # print(standard_deck[:10])