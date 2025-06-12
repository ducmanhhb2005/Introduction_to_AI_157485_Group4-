from collections import deque
from Overall.const import FOOD, EMPTY, WALL, MONSTER
import random

DDX = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def isValid(_map, row: int, col: int, N: int, M: int) -> bool:
    return 0 < row < N and 0 < col < M and (_map[row][col] == FOOD or _map[row][col] == EMPTY)


def isValid2(_map, row: int, col: int, N: int, M: int) -> bool:
    return 0 < row < N and 0 < col < M and _map[row][col] != WALL


def Manhattan(x1: int, y1: int, x2: int, y2: int) -> float:
    return abs(x1 - x2) + abs(y1 - y2)


def find_nearest_food(_map, N, M, start_row: int, start_col: int):
    visited = [[False for _ in range(M)] for _ in range(N)]
    trace = [[[-1, -1] for _ in range(M)] for _ in range(N)]

    lt = []
    visited[start_row][start_col] = True
    lt.append([start_row, start_col])

    chk = False

    while len(lt) > 0:
        [row, col] = lt.pop(0)
        if _map[row][col] == FOOD:
            chk = True
            break

        for [d_r, d_c] in DDX:
            new_row, new_col = row + d_r, col + d_c
            if isValid(_map, new_row, new_col, N, M) and not visited[new_row][new_col]:
                visited[new_row][new_col] = True
                lt.append([new_row, new_col])
                trace[new_row][new_col] = [row, col]

    if not chk:
        return [-1, -1]

    result = [[row, col]]
    [row, col] = trace[row][col]
    while row != -1:
        result.insert(0, [row, col])
        [row, col] = trace[row][col]
    # print(start_row, start_col, result)
    return result[-1]

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

def best_move(res, prev_row, prev_col):
    if len(res) > 0:
    # Lấy giá trị tốt nhất
        best_value = res[-1][1]
    
        # Lấy tất cả các lựa chọn có giá trị tốt nhất
        best_choices = [move for move, value in res if value == best_value]
        
        # Tránh vị trí trước đó nếu có thể
        non_prev_choices = [choice for choice in best_choices if choice != [prev_row, prev_col]]
        # print(non_prev_choices)
        if non_prev_choices:
            return random.choice(non_prev_choices)
        else:
            # Nếu tất cả đều là vị trí trước, chọn random từ tất cả
            return random.choice(best_choices)
    return [] 

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
    