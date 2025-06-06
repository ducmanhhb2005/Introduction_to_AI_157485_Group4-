from collections import deque
from Overall.constants import FOOD, EMPTY, WALL

DDX = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def isValid(_map, row: int, col: int, N: int, M: int) -> bool:
    return 0 < row < N and 0 < col < M and (_map[row][col] == FOOD or _map[row][col] == EMPTY)


def isValid2(_map, row: int, col: int, N: int, M: int) -> bool:
    return 0 < row < N and 0 < col < M and _map[row][col] != WALL


def Manhattan(x1: int, y1: int, x2: int, y2: int) -> float:
    return abs(x1 - x2) + abs(y1 - y2)


def find_nearest_food(_food_Position: list[list[int]], start_row: int, start_col: int):
    food_row, food_col, _id = -1, -1, -1
    for idx in range(len(_food_Position)):
        if food_row == -1:
            _id = idx
            [food_row, food_col] = _food_Position[idx]
        elif Manhattan(food_row, food_col, start_row, start_col) > Manhattan(_food_Position[idx][0],
                                                                             _food_Position[idx][1], start_row,
                                                                             start_col):
            _id = idx
            [food_row, food_col] = _food_Position[idx]

    return [food_row, food_col, _id]

def compute_all_pairs_shortest_paths(_map, N, M):
    """
    Returns a dictionary with keys ((row1, col1), (row2, col2)) and values as the shortest path distance between them.
    Only considers positions that are not WALL.
    """
    def bfs(start_row, start_col):
        distances = {}
        visited = [[False for _ in range(M)] for _ in range(N)]
        queue = deque()
        queue.append((start_row, start_col, 0))
        visited[start_row][start_col] = True
        while queue:
            row, col, dist = queue.popleft()
            distances[(row, col)] = dist
            for dr, dc in DDX:
                nr, nc = row + dr, col + dc
                if 0 <= nr < N and 0 <= nc < M and not visited[nr][nc] and _map[nr][nc] != WALL:
                    visited[nr][nc] = True
                    queue.append((nr, nc, dist + 1))
        return distances

    all_distances = {}
    for row in range(N):
        for col in range(M):
            if _map[row][col] != WALL:
                dists = bfs(row, col)
                for pos, dist in dists.items():
                    all_distances[((row, col), pos)] = dist
    return all_distances

class Counter(dict):
    def normalize(self):
        total = sum(self.values())
        if total == 0:
            return
        for key in self:
            self[key] /= total
              # Debugging line to check normalization 
    def choosefromdistribution(self):   
        import random

        r = random.random()
        upto = 0.0
        for key in self:
            if upto + self[key] >= r:
                return key
            upto += self[key]

    def __missing__(self, key):
        self[key] = 0
        return 0
