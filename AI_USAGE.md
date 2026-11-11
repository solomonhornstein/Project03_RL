# AI Usage Documentation

The Durham Academy Honor Code requires documenting any AI other than FlintK12. For this project I used **Claude** (Anthropic's AI assistant, claude.ai) throughout. This file records what Claude produced, what I did myself, and why I used it.

## Why I used Claude

I used Claude to help me plan, build, debug, and document this project within the two and a half weeks available. I wanted help turning the reinforcement learning ideas from class into working Python and Pygame code, setting up experiments to test my design choices, and getting past setup problems (the pygame install and Git errors) that would otherwise have cost days. I chose Claude over FlintK12 because I found it significantly more capable for this kind of work.

## What Claude produced

| Item | What Claude did |
|---|---|
| Project planning | Suggested a schedule, compared game options, and recommended tabular Q-learning over a neural network so every part would be explainable |
| `flappy_env.py` | Wrote the full game environment, including the state (dx, dy, velocity) and reward design (+0.1 per frame, +5 per pipe, −100 on crash) |
| `agent.py` | Wrote the Q-learning agent: state bucketing, epsilon-greedy policy, Q-update, save/load; later added saving bucket sizes with the Q-table |
| `train.py` | Wrote the training loop, stats panel, and learning graph; later added experiment flags (alpha, rewards, seed, run name) and the UP/DOWN speed control |
| `compare_runs.py` | Wrote the script that plots learning curves and prints the summary table |
| Experiment design | Suggested which experiments to run (bucket size, learning rate, reward changes), using fixed seeds, and rerunning with a second seed |
| Debugging help | Diagnosed the pygame install failure on Python 3.14 (fix: pygame-ce), the CRLF line-ending warning, and the Xcode license error blocking Git |
| Git guidance | Told me which files to commit, what to leave out (`.idea`, experiment Q-tables), and wrote commit messages |
| Documentation | Wrote the AI notes in each code file's header, `README.md` (including the "What the results show" analysis, based on my experiment results), and a draft of this file |

## What I did myself

- Chose Flappy Bird from the options Claude suggested
- Set up the PyCharm project, virtual environment, and GitHub repository
- Ran the code, played the game by hand to check it worked, and followed the fixes for install and Git problems
- Ran all training: the first training runs, the five controlled experiments, and the seed-2 confirmation runs
- Generated and checked the comparison charts, including regenerating the five-experiment chart after accidentally overwriting it
- Decided, based on my results, to change the default learning rate from 0.7 to 0.3 and to use the seed-1 α = 0.3 agent as the final demo agent
- Read and checked the README's results analysis against my charts and data

## How I checked the AI's work

- Played the game myself before training to confirm the physics and controls worked
- Watched the trained agent with `--watch` to confirm it actually learned
- Compared learning curves across experiments and seeds instead of trusting a single run