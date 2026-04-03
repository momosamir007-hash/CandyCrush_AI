import numpy as np
from typing import List, Dict, Tuple

class CandyEngine:
    def __init__(self, grid_matrix: np.ndarray):
        self.grid = grid_matrix
        self.rows, self.cols = self.grid.shape

    def find_all_moves(self) -> List[Dict]:
        valid_moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                current_candy = self.grid[r, c]
                if current_candy in ['empty', 'blocker', 'unknown']:
                    continue
                if c < self.cols - 1:
                    right_candy = self.grid[r, c + 1]
                    if right_candy not in ['empty', 'blocker', 'unknown'] and current_candy != right_candy:
                        score, match_details = self._evaluate_swap((r, c), (r, c + 1))
                        if score > 0:
                            valid_moves.append({
                                'pos1': (r, c),
                                'pos2': (r, c + 1),
                                'direction': 'Right ➡️',
                                'score': score,
                                'details': match_details
                            })
                if r < self.rows - 1:
                    down_candy = self.grid[r + 1, c]
                    if down_candy not in ['empty', 'blocker', 'unknown'] and current_candy != down_candy:
                        score, match_details = self._evaluate_swap((r, c), (r + 1, c))
                        if score > 0:
                            valid_moves.append({
                                'pos1': (r, c),
                                'pos2': (r + 1, c),
                                'direction': 'Down ⬇️',
                                'score': score,
                                'details': match_details
                            })
        valid_moves.sort(key=lambda x: x['score'], reverse=True)
        return valid_moves

    def _evaluate_swap(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> Tuple[int, str]:
        temp_grid = np.copy(self.grid)
        temp_grid[pos1[0], pos1[1]], temp_grid[pos2[0], pos2[1]] = \
            temp_grid[pos2[0], pos2[1]], temp_grid[pos1[0], pos1[1]]
        matches1 = self._check_matches_at(temp_grid, pos1[0], pos1[1])
        matches2 = self._check_matches_at(temp_grid, pos2[0], pos2[1])
        total_score = 0
        details = []
        for m in [matches1, matches2]:
            if m['h_len'] >= 3:
                total_score += self._calculate_score(m['h_len'])
                details.append("Horizontal " + str(m['h_len']))
            if m['v_len'] >= 3:
                total_score += self._calculate_score(m['v_len'])
                details.append("Vertical " + str(m['v_len']))
            if m['h_len'] >= 3 and m['v_len'] >= 3:
                total_score += 50
                details.append("Wrapped Candy!")
        if not details:
            return 0, "No match"
        return total_score, " | ".join(details)

    def _check_matches_at(self, grid: np.ndarray, r: int, c: int) -> Dict:
        color = grid[r, c]
        if color in ['empty', 'blocker', 'unknown']:
            return {'h_len': 0, 'v_len': 0}
        left = c
        while left > 0 and grid[r, left - 1] == color:
            left -= 1
        right = c
        while right < self.cols - 1 and grid[r, right + 1] == color:
            right += 1
        h_len = right - left + 1
        up = r
        while up > 0 and grid[up - 1, c] == color:
            up -= 1
        down = r
        while down < self.rows - 1 and grid[down + 1, c] == color:
            down += 1
        v_len = down - up + 1
        return {'h_len': h_len, 'v_len': v_len}

    def _calculate_score(self, match_length: int) -> int:
        if match_length == 3:
            return 10
        elif match_length == 4:
            return 30
        elif match_length >= 5:
            return 100
        return 0
