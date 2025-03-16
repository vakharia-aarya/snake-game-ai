# SnaKeRL 🐍🤖

_A Fun Reinforcement Learning Experiment!_

## Description

I built SnaKeRL as a fun way to learn **Deep Q-Learning**! 🧠 This isn’t meant to be a super-smart AI—just a project to explore reinforcement learning while watching a snake make some... interesting choices. It sometimes learns, sometimes fails spectacularly, and that’s all part of the fun! 🎮🐍

## Features

✅ Uses **Deep Q-Learning** to play Snake  
✅ **Live visualization** of the AI in action  
✅ **Score tracking** with a line graph to see its progress 📈  
✅ Game settings can be tweaked in `constants.py`  
✅ Built with **Pygame** for graphics and **PyTorch** for learning

## Setup & Installation 🛠️

You’ll need **Python 3.8+** to run this. Here’s how to set it up:

1. Clone the repository:
   ```sh
   git clone https://github.com/vakharia-aarya/snake-game-ai.git
   cd snake-game-ai
   ```
2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Agent 🏃💨

To start the AI and watch it play, run:

```sh
python agent.py
```

You’ll see the game running along with a **live graph** showing the AI’s scores over time.

## Changing the Game ⚙️

Want to tweak how the AI plays? Open `constants.py` and change things like game speed, rewards, or training parameters.

## What to Expect 🤔

- This **isn’t a serious AI project**—just me having fun with RL. Expect **weird moves and hilarious failures**. 😆
- The AI **doesn’t think far ahead**, so it sometimes traps itself and dies. 💀
- There’s always room for improvements, but for now, it’s all about learning and experimenting!
