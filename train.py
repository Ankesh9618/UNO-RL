# uno_project/train.py

import random
import torch
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Use absolute imports from the package
from uno_game.agent import UnoAgent
from uno_game.player import createPlayerList
from uno_game.game import ActiveGame

def train_agents(num_variants=10, num_players_per_game=4, num_training_games=5000, eval_interval=500, num_eval_games=100, save_interval=1000, model_dir="models"):
    """Trains multiple agent variants through self-play."""

    print(f"Starting training process...")
    print(f" - Agent Variants: {num_variants}")
    print(f" - Players per Game: {num_players_per_game}")
    print(f" - Training Games: {num_training_games}")
    print(f" - Evaluation Interval: {eval_interval} games")
    print(f" - Evaluation Games: {num_eval_games}")
    print(f" - Model Save Interval: {save_interval} games")
    print(f" - Model Save Directory: {model_dir}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")

    # --- Agent Initialization ---
    agents = [UnoAgent(agent_id=i, variant=i) for i in range(num_variants)]
    print(f"Initialized {len(agents)} agent variants.")

    # --- Tracking ---
    win_rates = defaultdict(list) # {variant: [win_rate_at_eval_1, ...]}
    avg_rewards = defaultdict(list) # {variant: [avg_reward_at_eval_1, ...]}
    game_numbers = [] # Store game number at evaluation points

    # --- Training Loop ---
    for game_num in range(1, num_training_games + 1):
        print(f"\n--- Starting Training Game {game_num}/{num_training_games} ---")

        # Select agents for this game (randomly chosen from variants)
        # Ensure enough unique agents if possible, otherwise allow duplicates
        selected_agent_indices = random.sample(range(num_variants), min(num_variants, num_players_per_game))
        # If fewer variants than players, duplicate randomly
        while len(selected_agent_indices) < num_players_per_game:
            selected_agent_indices.append(random.choice(range(num_variants)))
        random.shuffle(selected_agent_indices) # Shuffle assignment order

        current_game_agents = [agents[i] for i in selected_agent_indices]
        print(f"Selected agents for game {game_num}: {[f'V{a.variant}' for a in current_game_agents]}")


        # Setup game
        try:
             player_head = createPlayerList(num_players_per_game, current_game_agents)
        except ValueError as e:
             print(f"Error creating players: {e}. Skipping game.")
             continue

        game = ActiveGame(player_head, num_players_per_game, training_mode=True)

        # Run the game
        game_completed = game.run_game_loop(max_turns=500) # Use the loop runner

        if game_completed:
             print(f"Game {game_num} finished. Winner(s): {game.winners}")
             # Epsilon decay happens within agent.step(done=True)
        else:
             print(f"Game {game_num} failed to start or complete properly.")


        # --- Evaluation Phase ---
        if game_num % eval_interval == 0:
            print(f"\n--- Evaluating Agents at Game {game_num} ---")
            game_numbers.append(game_num)
            current_win_rates = evaluate_agents(agents, num_players_per_game, num_eval_games)

            for i, agent in enumerate(agents):
                variant = agent.variant
                wr = current_win_rates.get(variant, 0.0) # Get win rate for this variant
                win_rates[variant].append(wr)

                # Calculate average reward since last evaluation
                # Need to access agent's internal reward tracking
                # Simple placeholder: just report win rate for now
                # avg_rew = sum(agent.total_rewards_history[-eval_interval:]) / eval_interval if agent.total_rewards_history else 0
                # avg_rewards[variant].append(avg_rew)

                print(f"  Variant {variant}: Eval Win Rate = {wr:.2%}")
                # print(f"  Variant {variant}: Avg Reward (last {eval_interval} games) = {avg_rew:.2f}") # Needs proper tracking

            # Optional: Plot progress
            # plot_progress(game_numbers, win_rates, avg_rewards)


        # --- Save Models ---
        if game_num % save_interval == 0:
            print(f"\n--- Saving Agent Models at Game {game_num} ---")
            for agent in agents:
                model_path = os.path.join(model_dir, f"uno_agent_variant_{agent.variant}_game_{game_num}.pth")
                agent.save(model_path)

    print("\n--- Training Complete ---")
    # Final evaluation and saving
    print("--- Final Evaluation ---")
    final_win_rates = evaluate_agents(agents, num_players_per_game, num_eval_games)
    for i, agent in enumerate(agents):
         variant = agent.variant
         wr = final_win_rates.get(variant, 0.0)
         print(f"  Variant {variant}: Final Eval Win Rate = {wr:.2%}")

    print("--- Saving Final Models ---")
    for agent in agents:
         model_path = os.path.join(model_dir, f"uno_agent_variant_{agent.variant}_final.pth")
         agent.save(model_path)

    # Optional: Final plots
    # plot_progress(game_numbers + [num_training_games], win_rates, avg_rewards) # Add final point


def evaluate_agents(all_agents, num_players_per_game, num_eval_games):
    """Evaluates agents by playing games with exploration turned off."""
    print(f"Running {num_eval_games} evaluation games...")
    variant_wins = defaultdict(int)
    total_games_by_variant = defaultdict(int)

    original_epsilons = {a.variant: a.epsilon for a in all_agents}
    # Set agents to evaluation mode (low epsilon)
    for agent in all_agents:
        agent.epsilon = 0.0 # Greedy policy for evaluation
        agent.qnetwork.eval() # Set network to eval mode

    for i in range(num_eval_games):
        if (i+1) % (num_eval_games // 10) == 0: # Print progress
             print(f"  Eval game {i+1}/{num_eval_games}")

        # Select agents for eval game (similar to training)
        num_variants = len(all_agents)
        selected_agent_indices = random.sample(range(num_variants), min(num_variants, num_players_per_game))
        while len(selected_agent_indices) < num_players_per_game:
             selected_agent_indices.append(random.choice(range(num_variants)))
        random.shuffle(selected_agent_indices)
        current_eval_agents = [all_agents[idx] for idx in selected_agent_indices]

        # Track which variants are in this game
        variants_in_game = [a.variant for a in current_eval_agents]
        for v in variants_in_game:
             total_games_by_variant[v] += 1

        # Setup and run eval game (NO training)
        try:
            player_head = createPlayerList(num_players_per_game, current_eval_agents)
            eval_game = ActiveGame(player_head, num_players_per_game, training_mode=False) # IMPORTANT: Training mode OFF
            eval_game.run_game_loop(max_turns=500) # Run the game

            if eval_game.winners:
                winner_id = eval_game.winners[0]
                # Find the winning agent's variant
                for p_idx, p in enumerate(eval_game.get_all_players(include_winners=True)):
                     if p.playerId == winner_id:
                          if p.agent: # Ensure the player had an agent
                              winner_variant = p.agent.variant
                              variant_wins[winner_variant] += 1
                          break
        except Exception as e:
             print(f"Error during evaluation game {i+1}: {e}")


    # Restore original epsilons and training mode
    for agent in all_agents:
        agent.epsilon = original_epsilons[agent.variant]
        agent.qnetwork.train()

    # Calculate win rates
    win_rates = {v: (variant_wins[v] / total_games_by_variant[v]) if total_games_by_variant[v] > 0 else 0
                 for v in range(len(all_agents))}

    return win_rates

# Optional: Plotting function
def plot_progress(game_numbers, win_rates, avg_rewards):
     plt.figure(figsize=(12, 5))

     plt.subplot(1, 2, 1)
     for variant, rates in win_rates.items():
          if len(rates) == len(game_numbers):
               plt.plot(game_numbers, rates, label=f'Variant {variant}')
     plt.xlabel("Game Number")
     plt.ylabel("Evaluation Win Rate")
     plt.title("Agent Win Rates during Training")
     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
     plt.grid(True)

     # Add subplot for rewards if tracked properly
     # plt.subplot(1, 2, 2)
     # ... plot average rewards ...

     plt.tight_layout()
     plt.savefig("training_progress.png")
     print("Saved training progress plot to training_progress.png")
     # plt.show() # Uncomment to display plot immediately


# --- Main Execution ---
if __name__ == "__main__":
    # Configuration (adjust as needed)
    NUM_VARIANTS = 10 # Should match UnoAgent variants usually
    PLAYERS_PER_GAME = 4
    TOTAL_TRAINING_GAMES = 10000 # Increase for serious training
    EVAL_INTERVAL = 500      # Evaluate every 500 games
    NUM_EVAL_GAMES = 100       # Number of games per evaluation run
    SAVE_INTERVAL = 2000     # Save models every 2000 games
    MODEL_DIR = "uno_models"   # Directory to save models

    train_agents(
        num_variants=NUM_VARIANTS,
        num_players_per_game=PLAYERS_PER_GAME,
        num_training_games=TOTAL_TRAINING_GAMES,
        eval_interval=EVAL_INTERVAL,
        num_eval_games=NUM_EVAL_GAMES,
        save_interval=SAVE_INTERVAL,
        model_dir=MODEL_DIR
    )