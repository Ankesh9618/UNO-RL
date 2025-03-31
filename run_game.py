# run_game.py
# This script imports components from the uno_game package and runs the simulation.

# Use absolute imports from the package
from uno_game.player import createPlayerList
from uno_game.game import ActiveGame
import sys # To potentially get player count from command line

def main():
    # --- Game Configuration ---
    try:
        # Default to 5 players if no argument provided
        num_players = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        if num_players < 2:
            print("Error: Minimum 2 players required.")
            return
        print(f"Setting up UNO game for {num_players} players.")
    except ValueError:
        print("Error: Invalid number of players provided. Please enter an integer.")
        return
    except IndexError:
        # Should be caught by the default, but good practice
        num_players = 5
        print(f"Setting up UNO game for default {num_players} players.")

    initial_cards_per_player = 7 # Standard UNO rule
    max_simulation_turns = 500 # Prevent truly infinite loops in simulation

    # --- Setup ---
    try:
        player_head = createPlayerList(num_players)
    except ValueError as e:
        print(f"Error creating players: {e}")
        return

    active_game = ActiveGame(player_head, num_players)

    # --- Run ---
    active_game.run_game_loop(max_turns=max_simulation_turns)

if __name__ == "__main__":
    main() # Run the main function when the script is executed