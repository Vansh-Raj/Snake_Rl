import random
import math
from collections import deque
from typing import Tuple, List

import numpy as np
import pygame

from src.config import (
    GRID_SIZE, CELL_PIXELS, SNAKE_INIT_LEN,
    REWARD_FOOD, REWARD_DEAD, REWARD_STEP, REWARD_TOWARD_FOOD,
    RENDER_TRAIN, FPS_TRAIN, FPS_PLAY
)

# Directions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIR_VECS = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
ACTIONS = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}

class SnakeEnv:
    """
    - State: 11-dim vector (classic compact representation)
        [danger_left, danger_straight, danger_right,
         dir_up, dir_right, dir_down, dir_left,
         food_up, food_right, food_down, food_left]
    - Actions (discrete 3): 0=straight, 1=left turn, 2=right turn
    """

    def __init__(self, render=False, play_mode=False):
        self.grid = GRID_SIZE
        self.render_enabled = render
        self.play_mode = play_mode

        # Pygame setup only if rendering
        self.screen = None
        self.clock = None
        self.fps = FPS_PLAY if play_mode else FPS_TRAIN
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid * CELL_PIXELS, self.grid * CELL_PIXELS))
            pygame.display.set_caption("RL Snake")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        cx = self.grid // 2
        cy = self.grid // 2
        self.direction = random.choice([UP, RIGHT, DOWN, LEFT])
        # Build initial snake behind the head
        self.snake: deque[Tuple[int, int]] = deque()
        for i in range(SNAKE_INIT_LEN):
            dx, dy = DIR_VECS[(self.direction + 2) % 4]  # opposite direction to place body behind
            self.snake.appendleft((cx - dx * i, cy - dy * i))
        self.head = self.snake[0]
        self.spawn_food()

        self.score = 0
        self.steps_since_food = 0
        self.prev_distance_to_food = self._manhattan(self.head, self.food)

        return self.get_state()

    def spawn_food(self):
        while True:
            fx = random.randint(0, self.grid - 1)
            fy = random.randint(0, self.grid - 1)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                return

    def step(self, action: int):
        # Handle quit in render modes
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        self._apply_action(action)
        next_pos = self._next_head_pos(self.direction)

        reward = REWARD_STEP
        done = False

        # Collision?
        if self._is_collision(next_pos):
            reward = REWARD_DEAD
            done = True
            if self.render_enabled:
                self._render()
            return self.get_state(), reward, done, {"score": self.score}

        # Move snake
        self.snake.appendleft(next_pos)
        self.head = next_pos

        # Eat?
        if next_pos == self.food:
            self.score += 1
            reward += REWARD_FOOD
            self.spawn_food()
            self.steps_since_food = 0
        else:
            self.snake.pop()  # move tail
            self.steps_since_food += 1

        # Shaping: moved closer to food?
        dist = self._manhattan(self.head, self.food)
        if dist < self.prev_distance_to_food:
            reward += REWARD_TOWARD_FOOD
        self.prev_distance_to_food = dist

        if self.render_enabled:
            self._render()

        return self.get_state(), reward, done, {"score": self.score}

    def get_state(self) -> np.ndarray:
        head_x, head_y = self.head
        dir_up = self.direction == UP
        dir_right = self.direction == RIGHT
        dir_down = self.direction == DOWN
        dir_left = self.direction == LEFT

        # Left/straight/right relative positions
        left_dir = (self.direction - 1) % 4
        right_dir = (self.direction + 1) % 4
        front_dir = self.direction

        danger_left = self._is_collision(self._next_head_pos(left_dir))
        danger_straight = self._is_collision(self._next_head_pos(front_dir))
        danger_right = self._is_collision(self._next_head_pos(right_dir))

        food_up = self.food[1] < head_y
        food_right = self.food[0] > head_x
        food_down = self.food[1] > head_y
        food_left = self.food[0] < head_x

        state = np.array([
            float(danger_left),
            float(danger_straight),
            float(danger_right),
            float(dir_up),
            float(dir_right),
            float(dir_down),
            float(dir_left),
            float(food_up),
            float(food_right),
            float(food_down),
            float(food_left),
        ], dtype=np.float32)
        return state

    # --- helpers ---

    def _apply_action(self, action: int):
        if action == 1:   # left turn
            self.direction = (self.direction - 1) % 4
        elif action == 2: # right turn
            self.direction = (self.direction + 1) % 4
        # action 0: straight -> no change

    def _next_head_pos(self, direction: int):
        dx, dy = DIR_VECS[direction]
        hx, hy = self.head
        return (hx + dx, hy + dy)

    def _is_collision(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.grid or y < 0 or y >= self.grid:
            return True
        # body collision (ignore tail if we are moving into it while it moves away—handled by order in step)
        return pos in list(self.snake)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # --- rendering ---

    def _render(self):
        if not self.render_enabled:
            return
        self.clock.tick(self.fps)
        self.screen.fill((18, 18, 18))
        # draw food
        self._draw_cell(self.food, (200, 60, 60))
        # draw snake
        for i, p in enumerate(self.snake):
            color = (60, 200, 90) if i == 0 else (40, 160, 70)
            self._draw_cell(p, color)
        pygame.display.flip()

    def _draw_cell(self, pos: Tuple[int, int], color):
        x, y = pos
        r = pygame.Rect(x * CELL_PIXELS, y * CELL_PIXELS, CELL_PIXELS, CELL_PIXELS)
        pygame.draw.rect(self.screen, color, r)
