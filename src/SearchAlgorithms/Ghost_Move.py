from queue import PriorityQueue

from Overall.utils import *

Direction = {
    'UP': [0, 1],  
    'DOWN': [0, -1],        
    'LEFT': [-1, 0],
    'RIGHT': [1, 0],    
}

def reverse_direction(direction):
    reverse_map = {
        'UP': 'DOWN',
        'DOWN': 'UP',
        'LEFT': 'RIGHT',
        'RIGHT': 'LEFT'
    }
    return reverse_map.get(direction, None)

def Ghost_move_level4(_map, start_row, start_col, end_row, end_col, N, M):
    visited = [[False for _ in range(M)] for _ in range(N)]
    trace = {}
    cost = {}
    path = []
    queue = PriorityQueue()

    start = (start_row, start_col)
    end = (end_row, end_col)

    cost[(start_row, start_col)] = 0
    queue.put((Manhattan(start_row, start_col, end_row, end_col), start))

    while not queue.empty():
        current = queue.get()[1]
        visited[current[0]][current[1]] = True
        if current == end:
            path.append([current[0], current[1]])
            while current != start:
                current = trace[current]
                path.append([current[0], current[1]])
            path.reverse()
            return path[1] if len(path) > 1 else [start_row, start_col]

        for [d_r, d_c] in DDX:
            new_row, new_col = current[0] + d_r, current[1] + d_c
            if isValid2(_map, new_row, new_col, N, M) and not visited[new_row][new_col]:
                group = (new_row, new_col)
                cost[group] = cost[current] + 1
                queue.put((cost[group] + Manhattan(new_row, new_col, end_row, end_col), group))
                trace[group] = current

    return [start_row, start_col]

class RandomGhost:
    def __init__(self):
        self.counter = Counter()
    def move(self, _map, start_row, start_col, old_direction, N, M):
        moves = []
    
        for key, next in Direction.items():
            d_r, d_c = next
            new_row, new_col = start_row + d_r, start_col + d_c
            if isValid2(_map, new_row, new_col, N, M) :
                moves.append(key)  
        if not moves:
            return [start_row, start_col]
        
        for i in moves:
            self.counter[i] += 1
        if old_direction is not None and len(moves) > 1:
            reverse = reverse_direction(old_direction)
            self.counter[reverse] = 0
        self.counter.normalize()
        idx = self.counter.choosefromdistribution()
        [next_row, next_col] = start_row + Direction[idx][0], start_col + Direction[idx][1]
        return [next_row, next_col] if idx is not None else [start_row, start_col]

class DirectionalGhost:
    def __init__(self):
        self.counter = Counter()

    def move(self, _map, start_row, start_col, pac_row, pac_col, old_direction, N, M):
        proBest = 0.8
        moves = []

        for key, next in Direction.items():
            d_r, d_c = next
            new_row, new_col = start_row + d_r, start_col + d_c
            if isValid2(_map, new_row, new_col, N, M):
                moves.append(key)  
        if not moves:
            return [start_row, start_col]
        distances = [Manhattan(start_row + Direction[move][0], start_col + Direction[move][1], pac_row, pac_col) for move in moves]
        min_distance = min(distances)
        bestActions = [moves[i] for i in range(len(moves)) if distances[i] == min_distance]
        legal_moves = [moves[i] for i in range(len(moves)) if distances[i] != min_distance]
        for i in legal_moves:
            self.counter[i] += (1 - proBest) / len(legal_moves)
        for i in bestActions:
            self.counter[i] += proBest / len(bestActions)
        if old_direction is not None and len(moves) > 1:
            reverse = reverse_direction(old_direction)
            self.counter[reverse] = 0
        self.counter.normalize()
        idx = self.counter.choosefromdistribution()
        [next_row, next_col] = start_row + Direction[idx][0], start_col + Direction[idx][1]
        return [next_row, next_col] if idx is not None else [start_row, start_col]  