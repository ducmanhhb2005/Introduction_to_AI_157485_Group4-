from queue import PriorityQueue
from Overall.overall import find_nearest_food, Manhattan, DDX, isValid
import heapq

def heuristic(row, col, goal_row, goal_col, _ghost_positions):
    h = Manhattan(row, col, goal_row, goal_col)
    
    if _ghost_positions:
        ghost_penalty = 0
        min_ghost_dist = float('inf')
        
        for [g_r, g_c] in _ghost_positions:
            ghost_dist = Manhattan(row, col, g_r, g_c)
            min_ghost_dist = min(min_ghost_dist, ghost_dist)
            
            # More balanced ghost penalty - avoid infinity
            if ghost_dist == 0:
                ghost_penalty += 1000  # High penalty but not infinity
            elif ghost_dist <= 1:
                ghost_penalty += 100
            elif ghost_dist <= 2:
                ghost_penalty += 50
            elif ghost_dist <= 3:
                ghost_penalty += 20
        
        # Add additional penalty if too close to any ghost
        if min_ghost_dist <= 2:
            ghost_penalty += 50 / (min_ghost_dist + 0.1)
            
        h += ghost_penalty
    
    return h

def AStar(_map, _food_Position, _ghost_positions, start_row, start_col, N, M):
    if not _food_Position:
        return []
    
    # Find nearest food
    # print(f"ghost: {_ghost_positions}")
    [food_row, food_col] = find_nearest_food(_map, N, M, start_row, start_col)
    if food_row == -1:
        return []

    # Early termination if already at food
    if start_row == food_row and start_col == food_col:
        return [[start_row, start_col]]

    # Use more efficient data structures
    start = (start_row, start_col)
    end = (food_row, food_col)
    
    # Using heapq instead of PriorityQueue for better performance
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Use sets for faster lookup
    visited = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start_row, start_col, food_row, food_col, _ghost_positions)}
    
    # Keep track of nodes in open set for efficient updates
    in_open_set = {start}

    while open_set:
        # Get node with lowest f_score
        current_f, current = heapq.heappop(open_set)
        in_open_set.discard(current)
        
        # Skip if already visited
        if current in visited:
            continue
            
        visited.add(current)
        
        # Goal reached
        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append([current[0], current[1]])
                current = came_from[current]
            path.append([start_row, start_col])
            path.reverse()
            return path

        # Explore neighbors
        current_g = g_score[current]
        
        for [d_r, d_c] in DDX:
            new_row, new_col = current[0] + d_r, current[1] + d_c
            neighbor = (new_row, new_col)
            
            # Check if neighbor is valid and not visited
            if (not isValid(_map, new_row, new_col, N, M) or 
                neighbor in visited):
                continue
            
            # Calculate tentative g_score
            tentative_g = current_g + 1
            
            # If this path to neighbor is better than previous one
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(new_row, new_col, 
                                                          food_row, food_col, 
                                                          _ghost_positions)
                
                # Add to open set if not already there
                if neighbor not in in_open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    in_open_set.add(neighbor)

    return []
