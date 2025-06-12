from Overall.overall import DDX, isValid2, Manhattan
from Overall.const import FOOD, MONSTER, WALL, EMPTY
import random

MONSTER_POINT = [-100, -300, -500, -1000, -1e8]
FOOD_POINT = [5, 20, 50, 100, 200]

def point(depth, _type):
    if _type == FOOD:
        return FOOD_POINT[depth]

    elif _type == MONSTER:
        return MONSTER_POINT[depth]

def update_heuristic(_map, pacman_row, pacman_col, current_row, current_col, N, M, depth, _type, cost):
    visited = [[False for _ in range(M)] for _ in range(N)]
    
    visited[current_row][current_col] = True
    q = [[current_row, current_col, depth]]
    cost[current_row][current_col] += point(depth, _type)

    while len(q) > 0:
        [row, col, d] = q.pop(0)
        new_depth = d - 1

        if new_depth < 0:
            break

        for [d_r, d_c] in DDX:
            new_row, new_col = row + d_r, col + d_c
            if isValid2(_map, new_row, new_col, N, M) and not visited[new_row][new_col] and Manhattan(pacman_row, pacman_col, new_row, new_col) <= depth:
                cost[new_row][new_col] += point(new_depth, _type)
                # print(f"{current_row} {current_col}: {new_row} {new_col} point: {point(new_depth, _type)} sum: {cost[new_row][new_col]}")
                visited[new_row][new_col] = True
                q.append([new_row, new_col, new_depth])




def calc_heuristic(_map, start_row, start_col, N, M, depth, cost):
    visited = [[False for _ in range(M)] for _ in range(N)]
    
    visited[start_row][start_col] = True
    q = [[start_row, start_col, depth]]

    while len(q) > 0:
        [row, col, d] = q.pop(0)
        new_depth = d - 1

        if new_depth < 0:
            break

        for [d_r, d_c] in DDX:
            new_row, new_col = row + d_r, col + d_c
            if isValid2(_map, new_row, new_col, N, M) and not visited[new_row][new_col]:
                if _map[new_row][new_col] != EMPTY:
                    update_heuristic(_map, start_row, start_col, new_row, new_col, N, M, depth - 1, _map[new_row][new_col], cost)
                visited[new_row][new_col] = True
                q.append([new_row, new_col, new_depth])



def local_search(_map, prev_row, prev_col, start_row, start_col, N, M, _visited):
    cost = [[0 for _ in range(M)] for _ in range(N)]

    calc_heuristic(_map, start_row, start_col, N, M, 5, cost)

    max_f = -1e9

    result = []
    for [d_r, d_c] in DDX:
        new_row, new_col = start_row + d_r, start_col + d_c
        # print(new_row, new_col, cost[new_row][new_col])
        if isValid2(_map, new_row, new_col, N, M) and cost[new_row][new_col] - _visited[new_row][new_col] >= max_f and _map[new_row][new_col] != WALL:
            if cost[new_row][new_col] - _visited[new_row][new_col] == max_f:
                result.append([new_row, new_col])
            else:
                result.clear();
                result.append([new_row, new_col])
            max_f = cost[new_row][new_col] - _visited[new_row][new_col]

    # print(max_f)
    # print(result)
    # print(_map)
    if len(result) == 0:
        return []

    non_prev_positions = [pos for pos in result if pos != [prev_row, prev_col]]
    if non_prev_positions:
        return random.choice(non_prev_positions)
    else: 
        return random.choice(result)
