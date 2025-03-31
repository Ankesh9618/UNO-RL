import random
import torch
import os
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
from uno_game.agent import UnoAgent
from uno_game.player import createPlayerList
from uno_game.game import ActiveGame

def train_agents_with_gui():
    root = tk.Tk()
    root.title("UNO Training Progress")
    root.geometry("400x200")
    
    tk.Label(root, text="UNO Training Progress", font=("Arial", 14)).pack(pady=5)
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, length=300, variable=progress_var, maximum=100)
    progress_bar.pack(pady=10)
    
    status_label = tk.Label(root, text="Starting Training...", font=("Arial", 10))
    status_label.pack(pady=5)
    
    root.update()
    
    NUM_VARIANTS = 10
    PLAYERS_PER_GAME = 4
    TOTAL_TRAINING_GAMES = 10000
    EVAL_INTERVAL = 500
    MODEL_DIR = "uno_models"
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    agents = [UnoAgent(agent_id=i, variant=i) for i in range(NUM_VARIANTS)]
    
    for game_num in range(1, TOTAL_TRAINING_GAMES + 1):
        progress_var.set((game_num / TOTAL_TRAINING_GAMES) * 100)
        status_label.config(text=f"Training Game {game_num}/{TOTAL_TRAINING_GAMES}")
        root.update()
        
        selected_agents = random.sample(agents, PLAYERS_PER_GAME)
        player_head = createPlayerList(PLAYERS_PER_GAME, selected_agents)
        game = ActiveGame(player_head, PLAYERS_PER_GAME, training_mode=True)
        game.run_game_loop(max_turns=500)
        
        if game_num % EVAL_INTERVAL == 0:
            status_label.config(text=f"Evaluating at game {game_num}...")
            root.update()
            # Evaluation logic here (placeholder)
            
    status_label.config(text="Training Complete!")
    root.mainloop()

if __name__ == "__main__":
    train_agents_with_gui()
