from Overall.overall import DDX, isValid, FOOD

def Bfs(_map, prev_row, prev_col, start_row, start_col, N, M):
    visited = [[False for _ in range(M)] for _ in range(N)]
    trace = [[[-1, -1] for _ in range(M)] for _ in range(N)]

    lt = []
    visited[start_row][start_col] = True
    visited[prev_row][prev_col] = True
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
        return []

    result = [[row, col]]
    [row, col] = trace[row][col]
    while row != -1:
        result.insert(0, [row, col])
        [row, col] = trace[row][col]

    return result