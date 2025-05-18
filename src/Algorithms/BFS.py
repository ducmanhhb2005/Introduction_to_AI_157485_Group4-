from src.Overall.utils import find_nearest_food
from src.Overall.utils import isValid
from src.Overall.utils import isFood
from src.Overall.utils import DDX

def BFS(_map, _food_Position, start_row, start_col, N, M):
    visited = [[False for _ in range(M)] for _ in range(N)]
    trace = [[[-1, -1] for _ in range(M)] for _ in range(N)]

    [food_row, food_col, _id] = find_nearest_food(_food_Position, start_row, start_col, N, M)

    if _id == -1:
        return []
    
    list = []
    check = False
    visited[start_row][start_col] = True

    while(len(list) > 0):
        [row, col] = list.pop(0);

        if [row, col] == [food_row, food_col]:
            check = True
            break
    
        for [d_r, d_c] in DDX:
            [next_row, next_col] = [row + d_r, next_col + d_c]
            if isValid(_map, next_row, next_col, N, M) and not visited[next_row][next_col]:
                visited[next_row][next_col] = True
                list.append([next_row, next_col])
                trace[next_row][next_col] = [row, col]

    if not check:
        _food_Position.pop(_id)
        return BFS(_map, _food_Position, start_row, start_col, N, M)

    result = [[food_row, food_col]]
    [row, col] = trace[food_row][food_col]
    while row != -1:
        result.insert(0, [row, col])
        [row, col] = trace[row][col]

    return result


                

        