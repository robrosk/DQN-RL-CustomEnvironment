import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np


class Environment:
    """Configurable Frozen Lake environment with solvable random map generation."""

    def __init__(
        self,
        grid_size: int = 4,
        hole_prob: float = 0.2,
        slip_prob: float = 0.2,
        randomize_on_reset: bool = True,
        min_path_length_ratio: float = 1.25,
        seed: Optional[int] = None,
        max_generation_attempts: int = 500,
    ) -> None:
        """Initialize the environment.

        Args:
            grid_size: Size of the square grid.
            hole_prob: Base probability of placing a hole in each non-terminal cell.
            slip_prob: Probability of slipping to a random direction when acting.
            randomize_on_reset: Regenerate a new map on every reset when True.
            min_path_length_ratio: Minimum ratio between shortest path length and grid size
                to filter out trivially easy mazes.
            seed: Optional random seed for reproducibility.
            max_generation_attempts: Maximum number of attempts to sample a solvable grid.
        """

        self.grid_size = grid_size
        self.hole_prob = hole_prob
        self.slip_prob = slip_prob
        self.randomize_on_reset = randomize_on_reset
        self.min_path_length_ratio = min_path_length_ratio
        self.max_generation_attempts = max_generation_attempts
        self.random_state = np.random.RandomState(seed)

        self.state_grid = None
        self.state: Tuple[int, int] = (0, 0)
        self.done = False

        self.action_space = [0, 1, 2, 3]  # 0=left, 1=right, 2=up, 3=down
        self.n_actions = len(self.action_space)

        self._generate_grid()
        self.n_rows, self.n_cols = self.state_grid.shape
        self.n_states = self.n_rows * self.n_cols

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------
    def _generate_grid(self) -> None:
        """Generate a solvable random grid that satisfies difficulty constraints."""

        attempts = 0
        while attempts < self.max_generation_attempts:
            attempts += 1
            grid = self._sample_grid()
            shortest_path = self._shortest_path_length(grid)

            if shortest_path is None:
                continue

            min_required = math.ceil(self.min_path_length_ratio * self.grid_size)
            if shortest_path < min_required:
                continue

            self.state_grid = grid
            return

        raise RuntimeError("Failed to generate a solvable grid within the attempt limit.")

    def _sample_grid(self) -> np.ndarray:
        """Sample a single grid without validating solvability."""

        grid = np.ones((self.grid_size, self.grid_size), dtype=np.int8)
        grid[0, 0] = 0  # Start
        grid[-1, -1] = 3  # Goal

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in [(0, 0), (self.grid_size - 1, self.grid_size - 1)]:
                    continue

                if self.random_state.rand() < self.hole_prob:
                    grid[r, c] = 2  # Hole

        return grid

    def _shortest_path_length(self, grid: np.ndarray) -> Optional[int]:
        """Return the shortest path length from start to goal avoiding holes."""

        start, goal = (0, 0), (self.grid_size - 1, self.grid_size - 1)
        queue: deque = deque([(start, 0)])
        visited = {start}

        while queue:
            (r, c), dist = queue.popleft()
            if (r, c) == goal:
                return dist

            for nr, nc in self._neighbors(r, c):
                if grid[nr, nc] == 2:
                    continue
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))

        return None

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return valid neighbor coordinates for a cell."""

        candidates = [
            (r, c - 1),
            (r, c + 1),
            (r - 1, c),
            (r + 1, c),
        ]

        return [
            (nr, nc)
            for nr, nc in candidates
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size
        ]

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------
    def reset(self) -> Tuple[int, int]:
        """Reset the environment to the start state and optionally regenerate the map."""

        if self.randomize_on_reset or self.state_grid is None:
            self._generate_grid()
            self.n_rows, self.n_cols = self.state_grid.shape
            self.n_states = self.n_rows * self.n_cols

        self.state = (0, 0)
        self.done = False
        return self.state

    def step(self, action: int):
        """Take a step in the environment with stochastic slipping."""

        row, col = self.state
        if self.random_state.rand() < self.slip_prob:
            action = self.random_state.choice(self.action_space)

        if action == 0:
            new_state = (row, max(col - 1, 0))
        elif action == 1:
            new_state = (row, min(col + 1, self.n_cols - 1))
        elif action == 2:
            new_state = (max(row - 1, 0), col)
        elif action == 3:
            new_state = (min(row + 1, self.n_rows - 1), col)
        else:
            raise ValueError(f"Invalid action {action}")

        cell_value = self.state_grid[new_state]
        if cell_value == 3:
            reward, done = 10.0, True
        elif cell_value == 2:
            reward, done = -5.0, True
        else:
            reward, done = 0.0, False

        self.state = new_state
        self.done = done
        return action, new_state, reward, done

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def pos_to_one_hot(self, row: int, col: int) -> np.ndarray:
        idx = row * self.n_cols + col
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def state_to_one_hot(self, state: Tuple[int, int]) -> np.ndarray:
        row, col = state
        return self.pos_to_one_hot(row, col)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_text(self) -> str:
        """Return a human-readable text representation of the grid."""

        char_map = {0: "S", 1: "F", 2: "H", 3: "G"}
        result = ""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) == self.state:
                    result += " X "
                else:
                    result += f" {char_map[self.state_grid[r, c]]} "
            result += "\n"
        return result

    def render_pygame(self, cell_size: int = 80) -> None:
        """Render the environment using pygame.

        This method requires pygame to be installed. It is intended for
        occasional visualization and is not used during training.
        """

        import pygame

        colors = {
            0: (0, 102, 204),  # Start - blue
            1: (173, 216, 230),  # Frozen - light blue
            2: (30, 30, 30),  # Hole - dark
            3: (0, 200, 0),  # Goal - green
        }
        agent_color = (255, 165, 0)

        pygame.init()
        screen = pygame.display.set_mode((self.n_cols * cell_size, self.n_rows * cell_size))
        pygame.display.set_caption("Frozen Lake")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, colors[self.state_grid[r, c]], rect)
                pygame.draw.rect(screen, (255, 255, 255), rect, 2)

        ar, ac = self.state
        center = (ac * cell_size + cell_size // 2, ar * cell_size + cell_size // 2)
        pygame.draw.circle(screen, agent_color, center, cell_size // 3)

        pygame.display.flip()


if __name__ == "__main__":
    env = Environment()
    print("Initial map:\n", env.render_text())
    state = env.reset()
    print("After reset:", state)
