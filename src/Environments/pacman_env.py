# Source/Environments/pacman_env.py
import numpy as np
import random
import os
import pygame
from collections import deque


from Overall.const import (EMPTY, WALL, FOOD, MONSTER, SIZE_WALL, YELLOW, BLUE,
                           IMAGE_PACMAN, IMAGE_GHOST, MARGIN, BLACK, WHITE, RED,
                           WIDTH, HEIGHT, NUM_GHOSTS_FOR_STATE) # Thêm NUM_GHOSTS_FOR_STATE
from Object.Player import Player
from Object.Food import Food



class PacmanEnv:
    def __init__(self, map_file_path, max_steps_per_episode=2000, screen_surface_for_render=None): 
        self.map_file_path = map_file_path
        if not os.path.exists(self.map_file_path):
            raise FileNotFoundError(f"Không tìm thấy file map tại: {self.map_file_path}")

        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.screen_surface = screen_surface_for_render
        self.current_episode_num_for_render = 0
        self.N=0; self.M=0; self._map_data=[]; self.pacman=None
        self.ghosts_initial_pos=[]; # Ví dụ Vị trí đầu tiên của ghosts [(1, 5), (5, 1)]
        self.ghosts=[]   #Lưu các Player (ghost cũng là một Player)
        self.foods_initial_pos=[];   #Vị trí ban đầu của food
        #Một danh sách các tuple tọa độ (row, col).
        self.foods=[]  #Danh sách các food
        self.score=0; self.initial_food_count=0
        self.ACTION_MAP = {0:(-1,0), 
                           1:(1,0), 
                           2:(0,-1), 
                           3:(0,1)}
        self.action_space_n = len(self.ACTION_MAP)
        self.position_history = deque(maxlen=5) #hàng đợi chứa 5 vị trí gần nhất của Pacman
        #deque([(3,3), (3,4), (3,3), (3,4), (3,3)], maxlen=5)
        self.steps_since_last_food = 0          #số bước kể từ lần ăn thức ăn cuối
        self.visited_cells_this_episode = set()  #Các ô duy nhất mà Pacman đã đi qua
        #Ví dụ: {(1, 1), (1, 2), (2, 2)}. Nếu Pacman đi lại vào ô (1, 1), tập hợp sẽ không thay đổi. 

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        source_directory = os.path.abspath(os.path.join(current_file_dir, ".."))
        if IMAGE_PACMAN: self.pacman_img_path = os.path.join(source_directory, IMAGE_PACMAN[0])
        
        self.ghost_imgs_paths = []
        if IMAGE_GHOST:
            for ghost_path in IMAGE_GHOST: self.ghost_imgs_paths.append(os.path.join(source_directory, ghost_path))
        

        self._initialize_game_internals()
        self._state_dim = self._calculate_state_dim() 

        if self.screen_surface:
            try: self.font = pygame.font.SysFont('Arial', 20)
            except: self.font = None
        else: self.font = None

    def _read_map_file(self): 
        pacman_start_pos=None; ghost_starts=[]; food_locs=[]; map_grid=[]
        with open(self.map_file_path, "r") as f:
            N, M = map(int, f.readline().split())
            for r in range(N):
                row_data = list(map(int, f.readline().split()))
                map_grid.append(row_data)
                for c, cell_type in enumerate(row_data):
                    if cell_type == FOOD: food_locs.append((r,c))
                    elif cell_type == MONSTER: ghost_starts.append((r,c))
            pacman_start_pos = tuple(map(int, f.readline().split()))
        return N, M, map_grid, pacman_start_pos, ghost_starts, food_locs

    def _initialize_game_internals(self):
        self.N, self.M, self._map_data, pacman_pos, ghost_initial_pos, food_initial_pos = self._read_map_file()
        if self.screen_surface:
            map_pixel_width = self.M * SIZE_WALL; map_pixel_height = self.N * SIZE_WALL
            MARGIN["LEFT"] = max(0, (WIDTH - map_pixel_width) // 2)
            MARGIN["TOP"] = max(0, (HEIGHT - map_pixel_height) // 2)

        self.pacman = Player(pacman_pos[0], pacman_pos[1], self.pacman_img_path)
        self.position_history.clear(); self.position_history.append((self.pacman.row, self.pacman.col))
        self.visited_cells_this_episode.clear(); self.visited_cells_this_episode.add((self.pacman.row, self.pacman.col))
        self.steps_since_last_food = 0

        self.ghosts_initial_pos = list(ghost_initial_pos)
        self.ghosts = []
        # Chỉ tạo số lượng ma tối đa là NUM_GHOSTS_FOR_STATE hoặc số lượng có trong map, lấy min
        num_ghosts_to_create = min(len(self.ghosts_initial_pos), NUM_GHOSTS_FOR_STATE)
        for i in range(num_ghosts_to_create):
            pos = self.ghosts_initial_pos[i]
            ghost_image_path = self.ghost_imgs_paths[i % len(self.ghost_imgs_paths)]
            self.ghosts.append(Player(pos[0], pos[1], ghost_image_path))

        self.foods_initial_pos = list(food_initial_pos)
        self.foods = []
        for r_idx, c_idx in self.foods_initial_pos:
             self.foods.append(Food(r_idx, c_idx, SIZE_WALL // 2, SIZE_WALL // 2, YELLOW))
        self.initial_food_count = len(self.foods) if self.foods else 1
        self.score = 0; self.current_step = 0

    def _calculate_state_dim(self) -> int:
        pacman_pos_dim = 2 #tọa độ theo trục ngang và trục dọc của Pacman
        surrounding_cells_dim = 4 * 3 # 4 hướng, mỗi hướng: tường, food, ma 
        nearest_food_dim = 3  # delta_row, delta_col, distance
        remaining_food_ratio_dim = 1
        
        # Thông tin cho NUM_GHOSTS_FOR_STATE con ma
        ghost_info_dim_per_ghost = 3 # norm_pos_r, norm_pos_c, norm_dist_to_pacman
        total_ghost_info_dim = NUM_GHOSTS_FOR_STATE * ghost_info_dim_per_ghost
        
        return pacman_pos_dim + surrounding_cells_dim + nearest_food_dim + \
               remaining_food_ratio_dim + total_ghost_info_dim
               # 2 + 12 + 3 + 1 + (4*3=12) = 30 features
# state_vector = [
#     # --- Vị trí Pacman (2) ---
#Công thức: (pac_r / (N-1), pac_c / (M-1))
#     0.333, 0.444,

#     # --- Các ô xung quanh (12) ---
#[có tường?, có food?, có ma?]
#     # Lên          Xuống         Trái          Phải
#     1.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,

#     # --- Thức ăn gần nhất (3) ---
#delta_row chuẩn hóa: (food_r - pac_r) / N
#delta_col chuẩn hóa: (food_c - pac_c) / M
#khoảng cách chuẩn hóa (Manhattan dist) /n+m-2
#     -0.1, -0.2, 0.167,

#     # --- Tỷ lệ thức ăn còn lại (1) --- số food còn lại / số food ban đầu
#     0.1,

#     # --- Thông tin 4 con ma (12) --- ng tin: [vị trí r chuẩn hóa, vị trí c chuẩn hóa, khoảng cách chuẩn hóa(/m+n-2)]
#     # Ma 1 (G1)       Ma 2 (G2)        Ma 3 (padding)   Ma 4 (padding)
#     0.333, 0.667, 0.111,  0.889, 0.889, 0.5,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0
# ]
    @property
    def observation_space_dim(self): return self._state_dim

    def _get_state(self) -> np.ndarray:
        features = []
        pac_r, pac_c = self.pacman.getRC()

        # Vị trí Pac-Man (Chuẩn hóa)
        features.append(pac_r / (self.N - 1 if self.N > 1 else 1.0))
        features.append(pac_c / (self.M - 1 if self.M > 1 else 1.0))

        # Thông tin về 4 ô kề cận (Lên, Xuống, Trái, Phải)
        # Thứ tự: Lên, Xuống, Trái, Phải
        # Mỗi hướng: [is_wall, is_food, is_ghost]
        for dr, dc in self.ACTION_MAP.values():
            next_r, next_c = pac_r + dr, pac_c + dc
            is_wall_in_dir, is_food_in_dir, is_ghost_in_dir = 0.0, 0.0, 0.0
            if not (0 <= next_r < self.N and 0 <= next_c < self.M): is_wall_in_dir = 1.0
            elif self._map_data[next_r][next_c] == WALL: is_wall_in_dir = 1.0
            else:
                if any(f.row == next_r and f.col == next_c for f in self.foods): is_food_in_dir = 1.0
                if any(g.row == next_r and g.col == next_c for g in self.ghosts): is_ghost_in_dir = 1.0
            features.extend([is_wall_in_dir, is_food_in_dir, is_ghost_in_dir])

        # Thông tin về Thức Ăn Gần Nhất
        norm_delta_r_food, norm_delta_c_food, norm_dist_food = 0.0, 0.0, 1.0
        if self.foods:
            min_dist_manhattan = float('inf'); nearest_f_obj = None
            for food_obj in self.foods:
                dist_m = abs(pac_r - food_obj.row) + abs(pac_c - food_obj.col)
                if dist_m < min_dist_manhattan: min_dist_manhattan = dist_m; nearest_f_obj = food_obj
            if nearest_f_obj:
                norm_delta_r_food = (nearest_f_obj.row - pac_r) / self.N
                norm_delta_c_food = (nearest_f_obj.col - pac_c) / self.M
                norm_dist_food = min_dist_manhattan / (self.N + self.M - 2 if (self.N + self.M - 2) > 0 else 1.0)
        features.extend([norm_delta_r_food, norm_delta_c_food, norm_dist_food])

        # Tỷ lệ Thức Ăn Còn Lại
        features.append(len(self.foods) / self.initial_food_count if self.initial_food_count > 0 else 0.0)

        # Thông tin về NUM_GHOSTS_FOR_STATE con ma gần nhất 
        # Sắp xếp ma theo khoảng cách đến Pac-Man
        sorted_ghosts = sorted(self.ghosts, key=lambda g: abs(pac_r - g.row) + abs(pac_c - g.col))
        for i in range(NUM_GHOSTS_FOR_STATE):
            if i < len(sorted_ghosts):
                ghost = sorted_ghosts[i]
                features.append(ghost.row / (self.N - 1 if self.N > 1 else 1.0)) # Norm ghost_r
                features.append(ghost.col / (self.M - 1 if self.M > 1 else 1.0)) # Norm ghost_c
                dist_to_g = (abs(pac_r - ghost.row) + abs(pac_c - ghost.col)) / \
                            (self.N + self.M - 2 if (self.N + self.M - 2) > 0 else 1.0)
                features.append(dist_to_g) # Norm distance to this ghost
            else:
                # Nếu không đủ NUM_GHOSTS_FOR_STATE con ma, điền giá trị mặc định (ví dụ: ma ở rất xa)
                features.extend([1.0, 1.0, 1.0]) # Ví dụ: vị trí (N,M) và khoảng cách max

        return np.array(features, dtype=np.float32)

    def reset(self):
        self._initialize_game_internals()
        self.position_history.clear()
        self.steps_since_last_food = 0
        self.visited_cells_this_episode.clear()
        self.position_history.append((self.pacman.row, self.pacman.col))
        self.visited_cells_this_episode.add((self.pacman.row, self.pacman.col))
        return self._get_state()

    def _is_valid_position(self, r, c):
        return 0 <= r < self.N and 0 <= c < self.M and self._map_data[r][c] != WALL

    def step(self, action: int):
        self.current_step += 1
        reward = 0.0 # Bỏ hình phạt mỗi bước đi
        done = False
        food_eaten_this_step = False

        # Di chuyển Pacman
        d_row, d_col = self.ACTION_MAP[action]
        next_pac_r, next_pac_c = self.pacman.row + d_row, self.pacman.col + d_col
        if self._is_valid_position(next_pac_r, next_pac_c):
            self.pacman.setRC(next_pac_r, next_pac_c)

        # Cập nhật lịch sử vị trí Pacman và ô đã ghé thăm
        current_pos_tuple = (self.pacman.row, self.pacman.col)
        self.position_history.append(current_pos_tuple)

        EXPLORATION_BONUS = 0.1 # Nhỏ khi khám phá
        if current_pos_tuple not in self.visited_cells_this_episode:
            reward += EXPLORATION_BONUS
            self.visited_cells_this_episode.add(current_pos_tuple)

        #  Di chuyển Ghosts
        for ghost in self.ghosts:
            valid_moves = []
            for dr_g, dc_g in self.ACTION_MAP.values():
                if self._is_valid_position(ghost.row + dr_g, ghost.col + dc_g):
                    valid_moves.append((ghost.row + dr_g, ghost.col + dc_g))
            if valid_moves: ghost.setRC(*random.choice(valid_moves))

       
        PROXIMITY_THRESHOLD = 1 # Rất gần
        PROXIMITY_PENALTY = -2.0 # Phạt tương đối
        pac_r_curr, pac_c_curr = self.pacman.getRC()
        for ghost in self.ghosts:
            ghost_r_curr, ghost_c_curr = ghost.getRC()
            distance = abs(pac_r_curr - ghost_r_curr) + abs(pac_c_curr - ghost_c_curr)
            if 0 < distance <= PROXIMITY_THRESHOLD:
                reward += PROXIMITY_PENALTY
                break


        # Xử lý va chạm và tương tác
        # Pacman ăn Food
        for i in range(len(self.foods) - 1, -1, -1):
            if self.pacman.row == self.foods[i].row and self.pacman.col == self.foods[i].col:
                self.score += 10
                reward += 10.0 # Phần thưởng ăn food
                self.foods.pop(i)
                food_eaten_this_step = True # cập nhật cờ
                break

        # steps_since_last_food SAU KHI BIẾT food_eaten_this_step 
        if food_eaten_this_step:
            self.steps_since_last_food = 0 # Reset nếu ăn được food
        else:
            self.steps_since_last_food += 1 # Tăng nếu không ăn được food

        
        
        # Phạt khi đi lòng vòng không ăn được food
        STAGNATION_PENALTY = -0.2 # Phạt nhỏ hơn proximity
        if not food_eaten_this_step and len(self.position_history) == self.position_history.maxlen:
            if self.position_history.count(self.position_history[-1]) >= 3:
                reward += STAGNATION_PENALTY

        info_status = "playing"
        # b. Pacman bị Ghost bắt
        for ghost in self.ghosts:
            if self.pacman.row == ghost.row and self.pacman.col == ghost.col:
                self.score -= 150 
                reward -= 150.0
                done = True; info_status = "lost"; break
        
        if not done:
            if not self.foods: # Thắng
                self.score += 1000 # Tăng mạnh thưởng thắng
                reward += 1000.0
                done = True; info_status = "won"
            elif self.current_step >= self.max_steps_per_episode:
                reward -= 50.0 # Phạt vừa phải khi hết giờ
                done = True; info_status = "max_steps_reached"
        
        return self._get_state(), reward, done, {"score":self.score, "steps":self.current_step, "status":info_status}

    def render(self, mode='text'): 
        if mode == 'pygame' and self.screen_surface:
            self.screen_surface.fill(BLACK)
            margin_left_map = MARGIN["LEFT"]; margin_top_map = MARGIN["TOP"]
            for r in range(self.N):
                
                for c in range(self.M):
                    if self._map_data[r][c] == WALL: pygame.draw.rect(self.screen_surface, BLUE, pygame.Rect(margin_left_map + c*SIZE_WALL, margin_top_map + r*SIZE_WALL, SIZE_WALL, SIZE_WALL))
            for food_obj in self.foods:
                food_obj.rect.topleft = (margin_left_map + food_obj.col*SIZE_WALL + (SIZE_WALL-food_obj.rect.width)//2, margin_top_map + food_obj.row*SIZE_WALL + (SIZE_WALL-food_obj.rect.height)//2)
                food_obj.draw(self.screen_surface)
            for ghost_obj in self.ghosts: ghost_obj.draw(self.screen_surface)
            if self.pacman: self.pacman.draw(self.screen_surface)
            if self.font:
                score_text = self.font.render(f"Score: {self.score} Steps: {self.current_step}/{self.max_steps_per_episode}", True, WHITE)
                self.screen_surface.blit(score_text, (10, 10))
                if self.current_episode_num_for_render > 0:
                    ep_text = self.font.render(f"Episode: {self.current_episode_num_for_render}", True, WHITE)
                    self.screen_surface.blit(ep_text, (10, 35))
        elif mode == 'text': 
            grid=[[' ']*self.M for _ in range(self.N)];_map_data=self._map_data
            for r in range(self.N):
                for c in range(self.M):
                    if _map_data[r][c]==WALL:grid[r][c]='#'
            for f in self.foods:grid[f.row][f.col]='.'
            for g in self.ghosts:grid[g.row][g.col]='G'
            if self.pacman:grid[self.pacman.row][self.pacman.col]='X' if any(g.row==self.pacman.row and g.col==self.pacman.col for g in self.ghosts) else 'P'
            ep_s=f" (Ep: {self.current_episode_num_for_render})" if self.current_episode_num_for_render>0 else ""
            print(f"--- Step: {self.current_step}, Score: {self.score}{ep_s} ---")
            for r_p in range(self.N):print(" ".join(grid[r_p]))
            print("-"*(self.M*2))


    def set_current_episode_num_for_render(self, ep_num: int):
        self.current_episode_num_for_render = ep_num

    def close(self):
        print("PacmanEnv: close() được gọi.")

if __name__ == '__main__':

    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root_dir = os.path.abspath(os.path.join(current_script_dir, "../../"))
    # example_map_path = os.path.join(project_root_dir, "Input", "Level1", "map4.txt")
    # env = PacmanEnv(map_file_path=example_map_path)
    # print(f"Calculated state_dim: {env.observation_space_dim}")
    # state_example = env.reset()
    # print(f"Example state (shape: {state_example.shape}):\n{state_example}")
    pass