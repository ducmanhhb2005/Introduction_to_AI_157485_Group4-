from Overall.utils import DDX
from Overall.utils import isValid2
from Overall.const import FOOD
from Overall.const import MONSTER
from Overall.const import WALL



def update_heuristic(_map, start_row, start_col, current_row, current_col, N, M, depth, visited, _type, cost):
    visited.append((current_row, current_col))

    if depth < 0:
        return
    
    if (start_row, start_col) == (current_row, current_col):
        return
    
    point = 0
    if _type == FOOD:
        if depth == 2:
            point = 35
        if depth == 1:
            point = 10
        if depth == 0:
            point = 5

    elif _type == MONSTER:
        if depth == 2:
            point = float("-inf")
        if depth == 1:
            point = float("-inf")
        if depth == 0:
            point = -100

    cost[current_row][current_col] += point

    for [d_r, d_c] in DDX:
        next_row, next_col = current_row + d_r, current_col + d_c
        if isValid2(next_row, next_col) and (next_row, next_col) not in visited:
            update_heuristic(_map, start_row, start_col, next_row, next_col, N, M, depth - 1, visited.copy(), _type, cost)

def calc_heuristic(_map, start_row, start_col, current_row, current_col, N, M, depth, visited, cost):
    visited.append((current_row, current_col))

    if depth <= 0:
        return
    
    for [d_r, d_c] in DDX:
        next_row, next_col = current_row + d_r, current_col + d_c
        if isValid2(next_row, next_col) and (next_row, next_col) not in visited:
            sub_visited = []

            if _map[next_row][next_col] == FOOD:
                update_heuristic(_map, start_row, start_col, N, M, 2, sub_visited, FOOD, cost)
            elif _map[next_row][next_col] == MONSTER:
                update_heuristic(_map, start_row, start_col, N, M, 2, sub_visited, MONSTER, cost)

            calc_heuristic(_map, start_row, start_col, next_row, next_col, N, M, depth - 1, visited, cost)


def local_search(_map, start_row, start_col, N, M, _visited):
    visited = []
    cost = [[0 for _ in range(M)] for _ in range(N)]

    calc_heuristic(_map, start_row, start_col, start_row, start_col, N, M, 3, visited, cost)

    max_f = float("-inf")

    result = []

    for [d_r, d_c] in DDX:
        next_row, next_col = start_row + d_r, start_col + d_c
        if cost[next_row][next_col] - _visited[next_row][next_col] > max_f and _map[next_row][next_col] != WALL:
            max_f = cost[next_row][next_col] - _visited[next_row][next_col] 
            result = [next_row, next_col]
        
    return result




