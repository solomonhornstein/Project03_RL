"""
train.py -- runs the agent-environment loop for Project03_RL.

Usage:
  python train.py              train from scratch (window shows the game + stats)
  python train.py --resume     keep training from the saved q_table.pkl
  python train.py --watch      load q_table.pkl and watch the trained agent (no exploring)

Keys while running:
  R   toggle rendering (OFF = trains far faster; the stats panel still updates)
  S   save the Q-table now
  ESC quit (the Q-table is saved automatically)
"""

import argparse
import csv
import pygame

from flappy_env import FlappyEnv, WIDTH, HEIGHT
from agent import QAgent

PANEL_WIDTH = 320
MAX_SCORE = 500          # end an episode here so a skilled agent doesn't run forever
SAVE_EVERY = 500         # episodes between automatic saves
Q_PATH = "q_table.pkl"
LOG_PATH = "scores.csv"


def draw_panel(screen, font, small, agent, episode, scores, best, rendering, watch):
    """Right-hand panel: training stats plus a graph of the moving-average score."""
    x0 = WIDTH
    pygame.draw.rect(screen, (25, 25, 35), (x0, 0, PANEL_WIDTH, HEIGHT))

    recent = scores[-50:]
    avg = sum(recent) / len(recent) if recent else 0
    lines = [
        "WATCH MODE" if watch else "TRAINING",
        f"Episode: {episode}",
        f"Epsilon: {agent.epsilon:.4f}",
        f"Last score: {scores[-1] if scores else 0}",
        f"Best score: {best}",
        f"Avg (last 50): {avg:.1f}",
        f"States seen: {len(agent.q)}",
        f"Rendering: {'ON' if rendering else 'OFF (R)'}",
    ]
    for i, text in enumerate(lines):
        screen.blit(font.render(text, True, (230, 230, 230)), (x0 + 15, 15 + i * 30))

    # Graph of the 50-episode moving average across all of training
    gx, gy, gw, gh = x0 + 15, 290, PANEL_WIDTH - 30, 260
    pygame.draw.rect(screen, (60, 60, 75), (gx, gy, gw, gh), 1)
    screen.blit(small.render("Avg score (50-ep window)", True, (180, 180, 180)), (gx, gy - 20))
    if len(scores) >= 2:
        moving = []
        running = 0
        for i, s in enumerate(scores):
            running += s
            if i >= 50:
                running -= scores[i - 50]
            moving.append(running / min(i + 1, 50))
        # Sample down to the graph's pixel width
        step = max(1, len(moving) // gw)
        pts_vals = moving[::step]
        top = max(max(pts_vals), 1)
        points = [(gx + i * gw / max(len(pts_vals) - 1, 1),
                   gy + gh - (v / top) * (gh - 5)) for i, v in enumerate(pts_vals)]
        if len(points) >= 2:
            pygame.draw.lines(screen, (90, 200, 120), False, points, 2)
        screen.blit(small.render(f"max {top:.1f}", True, (180, 180, 180)), (gx + 5, gy + 5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL_WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy RL -- Q-learning")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    small = pygame.font.SysFont(None, 20)

    env = FlappyEnv()
    agent = QAgent()
    if args.resume or args.watch:
        agent.load(Q_PATH)

    rendering = args.watch     # training starts fast; watch mode always renders
    scores, best = [], 0
    running = True

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        while not done and env.score < MAX_SCORE:
            # --- handle keys (every frame when rendering, occasionally when not) ---
            if rendering or env.frames % 200 == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        rendering = not rendering
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                        agent.save(Q_PATH)
            if not running:
                break

            # --- the agent-environment loop ---
            action = agent.choose_action(state, greedy=args.watch)
            next_state, reward, done = env.step(action)
            if not args.watch:
                agent.learn(state, action, reward, next_state, done)
            state = next_state

            if rendering:
                env.render(screen, font)
                draw_panel(screen, font, small, agent, episode, scores, best, rendering, args.watch)
                pygame.display.flip()
                clock.tick(60)

        if not running:
            break

        if not args.watch:
            agent.end_episode()
        scores.append(env.score)
        best = max(best, env.score)

        # Refresh the panel between episodes even when rendering is off
        if not rendering and episode % 10 == 0:
            env.render(screen, font)
            draw_panel(screen, font, small, agent, episode, scores, best, rendering, args.watch)
            pygame.display.flip()

        if not args.watch and episode % SAVE_EVERY == 0:
            agent.save(Q_PATH)

    if not args.watch:
        agent.save(Q_PATH)
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "score"])
            writer.writerows(enumerate(scores, start=1))
        print(f"Saved {Q_PATH} and {LOG_PATH}. Best score: {best}")
    pygame.quit()


if __name__ == "__main__":
    main()