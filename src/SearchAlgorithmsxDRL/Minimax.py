from Overall.overall import DDX, isValid, isValid2, compute_all_pairs_shortest_paths, best_move
from Overall.const import FOOD, MONSTER, EMPTY

_food_pos = []

def evaluationFunction(_map, _ghost, pac_row, pac_col, N, M, score):
    # get food position
    distancesToFoodList = []
    for (row, col) in _food_pos:
        distancesToFoodList.append(all_distance[(row, col), (pac_row, pac_col)])

    # Consts
    INF = 100000000.0  # Infinite value
    WEIGHT_FOOD = 40.0  # Food base value
    WEIGHT_GHOST = -150.0  # Ghost base value

    _score = score
    if len(distancesToFoodList) > 0:
        _score += WEIGHT_FOOD / (min(distancesToFoodList) if min(distancesToFoodList) != 0 else 1)
    else:
        _score += WEIGHT_FOOD

    for [g_r, g_c] in _ghost:
        distance = all_distance[(pac_row, pac_col), (g_r, g_c)]
        if distance > 0:
            _score += WEIGHT_GHOST / distance
        else:
            return -INF

    return _score


def minimaxAgent(_map, _dis, prev_row, prev_col, pac_row, pac_col, N, M, depth, Score):
    def terminal(_map, _pac_row, _pac_col, _N, _M, _depth) -> bool:
        if _map[_pac_row][_pac_col] == MONSTER or _depth == 0:
            return True

        if len(_food_pos) > 0:
            return False

        return True

    def minimax(_map, _ghost, _pac_row, _pac_col, _N, _M, _depth, score, agent):
        global _food_num
        if terminal(_map, _pac_row, _pac_col, _N, _M, _depth):
            return evaluationFunction(_map, _ghost, _pac_row, _pac_col, _N, _M, score)

        if agent == -1:
            v = float("-inf")
        else:
            v = float("inf")

        if agent == -1:  # pacman move
            for [_d_r, _d_c] in DDX:
                _new_r, _new_c = _pac_row + _d_r, _pac_col + _d_c
                if isValid(_map, _new_r, _new_c, _N, _M):
                    state = _map[_new_r][_new_c]
                    _map[_new_r][_new_c] = EMPTY
                    if state == FOOD:
                        score += 20
                        _food_pos.pop(_food_pos.index((_new_r, _new_c)))
                        _food_num -= 1
                    else:
                        score -= 1
                    v = max(v, minimax(_map, _ghost, _new_r, _new_c, _N, _M, _depth, score, 0))
                    _map[_new_r][_new_c] = state
                    if state == FOOD:
                        score -= 20
                        _food_pos.append((_new_r, _new_c))
                        _food_num += 1
                    else:
                        score += 1

            return v

        pp = 0

        if (agent == len(_ghost)):
            return minimax(_map, _ghost, _pac_row, _pac_col, _N, _M, _depth - 1, score, -1)

        nextAgent = agent + 1
        if nextAgent == len(_ghost):
            nextAgent = -1

        if nextAgent == -1:
            _depth -= 1
        
        [g_r, g_c] = _ghost[agent]
        for [_d_r, _d_c] in DDX:
            _new_r, _new_c = g_r + _d_r, g_c + _d_c
            if isValid2(_map, _new_r, _new_c, _N, _M):
                pp += 1

        p = 1.0 / pp

        for [_d_r, _d_c] in DDX:
            _new_r, _new_c = g_r + _d_r, g_c + _d_c
            if isValid2(_map, _new_r, _new_c, _N, _M):
                state = _map[_new_r][_new_c]
                _map[_new_r][_new_c] = MONSTER
                _map[g_r][g_c] = EMPTY
                _ghost[agent][0], _ghost[agent][1] = _new_r, _new_c
                v =min(v,  minimax(_map, _ghost, _pac_row, _pac_col, _N, _M, _depth, score, nextAgent))
                _ghost[agent][0], _ghost[agent][1] = g_r, g_c
                _map[_new_r][_new_c] = state
                _map[g_r][g_c] = MONSTER
        return v

    global all_distance
    all_distance = compute_all_pairs_shortest_paths(_map, N, M)

    res = []
    global _food_pos
    global _food_num
    _food_num = 0
    _food_pos = []
    _ghost = []
    for _row in range(N):
        for _col in range(M):
            if _map[_row][_col] == FOOD:
                _food_pos.append((_row, _col))
                _food_num += 1
            if _map[_row][_col] == MONSTER:
                _ghost.append([_row, _col])

    for [d_r, d_c] in DDX:
        new_r, new_c = pac_row + d_r, pac_col + d_c
        if isValid(_map, new_r, new_c, N, M):
            _state = _map[new_r][new_c]
            _map[new_r][new_c] = EMPTY
            if _state == FOOD:
                Score += 20
                _food_pos.pop(_food_pos.index((new_r, new_c)))
                _food_num -= 1
            else:
                Score -= 1
            res.append(([new_r, new_c], minimax(_map, _ghost, new_r, new_c, N, M, depth, Score, -1)))
            _map[new_r][new_c] = _state
            if _state == FOOD:
                Score -= 20
                _food_pos.append((new_r, new_c))
                _food_num += 1
            else:
                Score += 1
    
    res.sort(key=lambda k: k[1])

    return best_move(res, prev_row, prev_col)