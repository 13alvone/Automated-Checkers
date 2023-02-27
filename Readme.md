# Automated Checkers

Board Lord is a Python program that implements a command-line interface for playing checkers. It allows two players to take turns moving pieces on a standard 8x8 checkerboard until one player has no moves left, or all of their pieces have been captured. 

## Usage

1. Clone this repository to your local machine.
2. Run the `checker_data_generator.py` script using Python from the command line:

```shell
python checker_data_generator.py
```

### Extended Usage

```Shell
usage: checker_data_generator.py [-h] [-n GAME_COUNT] [-p] [-s]

options:
  -h, --help            show this help message and exit
  -n GAME_COUNT, --game_count GAME_COUNT
                        How many games to run.
  -p, --print_board     Print Final Board
  -s, --silent          Print Only Final Summary
```

3. Follow the prompts in the terminal to make your moves.

## Features

- The board is printed to the console after every move.
- The program checks if a move is legal and updates the board accordingly.
- If a player has a jump available, they must take it.
- The game ends when one player has no legal moves or all of their pieces have been captured.
- The program keeps track of the number of moves made by each player and the time the game was played.
- The program stores each move in an SQLite database with a unique `move_id` for each move and a `game_id` that is constant throughout the game.

## Requirements

- Python3
- sqlite3

## Classes

- Checker: Defines a checkers piece.
- Board: Defines the checkers board and game mechanics.
- CheckersGame: Implements the gameplay loop.
