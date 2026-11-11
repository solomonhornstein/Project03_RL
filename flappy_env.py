"""
flappy_env.py -- the game ENVIRONMENT for Project03_RL.

Key RL idea: the agent never reads this file's rules. It only sees what
step() hands back -- a state, a reward, and whether the episode ended.
Everything about gravity, pipes and collisions is "hidden physics" the
agent has to discover through trial and error.

Run this file directly to play the game yourself (SPACE = flap).

AI USE (required by the DA Honor Code):
The initial version of this file was written by Claude (Anthropic's AI)
at my request, including the state and reward design. I ran, tested and
played it. See AI_USAGE.md for details.
"""

import random
import pygame

# ---------------- Game constants ----------------
WIDTH, HEIGHT = 400, 600
BIRD_X = 80               # bird stays at a fixed x; the pipes move toward it
BIRD_RADIUS = 12
GRAVITY = 0.5             # added to vertical velocity every frame
FLAP_VELOCITY = -8.0      # a flap sets velocity to this (negative = up)
MAX_FALL_SPEED = 10.0
PIPE_WIDTH = 60
PIPE_GAP = 160            # height of the opening between top and bottom pipe
PIPE_SPEED = 3            # pixels per frame
PIPE_SPACING = 220        # horizontal distance between consecutive pipes
GAP_MARGIN = 100          # keeps gaps away from the very top/bottom

# ---------------- Rewards (tune these later!) ----------------
REWARD_ALIVE = 0.1        # small reward for every frame survived
REWARD_PASS_PIPE = 5.0    # bonus for clearing a pipe
REWARD_DEATH = -100.0     # big penalty for crashing

# Actions the agent can choose
ACTION_NOTHING = 0
ACTION_FLAP = 1


class Pipe:
    """One pair of pipes (top and bottom) with a gap between them."""

    def __init__(self, x):
        self.x = x
        half_gap = PIPE_GAP // 2
        self.gap_center = random.randint(GAP_MARGIN + half_gap,
                                         HEIGHT - GAP_MARGIN - half_gap)
        self.passed = False

    @property
    def gap_top(self):
        return self.gap_center - PIPE_GAP / 2

    @property
    def gap_bottom(self):
        return self.gap_center + PIPE_GAP / 2

    def move(self):
        self.x -= PIPE_SPEED

    def collides_with(self, bird_y):
        # No collision possible unless the bird horizontally overlaps the pipe
        if BIRD_X + BIRD_RADIUS < self.x or BIRD_X - BIRD_RADIUS > self.x + PIPE_WIDTH:
            return False
        # Overlapping horizontally: crash if the bird pokes outside the gap
        return bird_y - BIRD_RADIUS < self.gap_top or bird_y + BIRD_RADIUS > self.gap_bottom


class FlappyEnv:
    """The environment. Same interface idea as OpenAI Gym: reset() and step()."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Start a new episode and return the starting state."""
        self.bird_y = HEIGHT / 2
        self.bird_vel = 0.0
        self.pipes = [Pipe(WIDTH + 50)]
        self.score = 0
        self.frames = 0
        self.done = False
        return self.get_state()

    def _next_pipe(self):
        """The first pipe the bird has not fully cleared yet."""
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH >= BIRD_X - BIRD_RADIUS:
                return pipe
        return self.pipes[-1]

    def get_state(self):
        """
        What the agent gets to 'see'. Raw numbers -- the agent decides how to
        bucket them. Three values:
          dx  = horizontal distance to the far edge of the next pipe
          dy  = bird height relative to the gap center (+ = below, - = above)
          vel = bird's vertical velocity (+ = falling)
        """
        pipe = self._next_pipe()
        dx = pipe.x + PIPE_WIDTH - BIRD_X
        dy = self.bird_y - pipe.gap_center
        return (dx, dy, self.bird_vel)

    def step(self, action):
        """Advance one frame. Returns (next_state, reward, done)."""
        if self.done:
            raise RuntimeError("Episode is over -- call reset() first.")

        # 1. Apply the action, then gravity
        if action == ACTION_FLAP:
            self.bird_vel = FLAP_VELOCITY
        self.bird_vel = min(self.bird_vel + GRAVITY, MAX_FALL_SPEED)
        self.bird_y += self.bird_vel

        # 2. Move pipes, spawn new ones, delete ones that left the screen
        for pipe in self.pipes:
            pipe.move()
        if self.pipes[-1].x < WIDTH - PIPE_SPACING:
            self.pipes.append(Pipe(WIDTH))
        self.pipes = [p for p in self.pipes if p.x + PIPE_WIDTH > 0]

        # 3. Compute the reward
        reward = REWARD_ALIVE
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X - BIRD_RADIUS:
                pipe.passed = True
                self.score += 1
                reward += REWARD_PASS_PIPE

        # 4. Check for death (hitting the ceiling, floor, or a pipe)
        hit_edge = self.bird_y - BIRD_RADIUS < 0 or self.bird_y + BIRD_RADIUS > HEIGHT
        hit_pipe = any(p.collides_with(self.bird_y) for p in self.pipes)
        if hit_edge or hit_pipe:
            self.done = True
            reward = REWARD_DEATH

        self.frames += 1
        return self.get_state(), reward, self.done

    def render(self, screen, font):
        """Draw the current frame. Training can skip this to run much faster."""
        screen.fill((135, 206, 235))  # sky
        for p in self.pipes:
            pygame.draw.rect(screen, (40, 160, 40), (p.x, 0, PIPE_WIDTH, p.gap_top))
            pygame.draw.rect(screen, (40, 160, 40),
                             (p.x, p.gap_bottom, PIPE_WIDTH, HEIGHT - p.gap_bottom))
        pygame.draw.circle(screen, (255, 215, 0), (BIRD_X, int(self.bird_y)), BIRD_RADIUS)
        screen.blit(font.render(f"Score: {self.score}", True, (0, 0, 0)), (10, 10))


def play_human():
    """Play the game yourself to check that the physics feel right."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy RL -- Human Mode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)
    env = FlappyEnv()

    running = True
    while running:
        action = ACTION_NOTHING
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if env.done:
                    env.reset()
                else:
                    action = ACTION_FLAP

        if not env.done:
            env.step(action)

        env.render(screen, font)
        if env.done:
            msg = font.render("Game over -- SPACE to restart", True, (200, 0, 0))
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    play_human()