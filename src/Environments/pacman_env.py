# Source/Environments/pacman_env.py
import numpy as np
import random
import os
import pygame

try:
    # Giả định constants.py và Object/ nằm trong thư mục Source/
    # và PacmanEnv.py nằm trong Source/Environments/
    # Khi train_agent.py đã thêm Source/ vào sys.path, các import này sẽ hoạt động
    from Overall.constants import (EMPTY, WALL, FOOD, MONSTER, SIZE_WALL, YELLOW, BLUE,
                           IMAGE_PACMAN, IMAGE_GHOST, MARGIN, BLACK, WHITE, RED,
                           WIDTH as SCREEN_WIDTH, HEIGHT as SCREEN_HEIGHT)
    from Object.Player import Player
    from Object.Food import Food
except ImportError as e:
    print(f"LỖI IMPORT TRONG PACMAN_ENV.PY: {e}")
    print("Đảm bảo Source/ nằm trong sys.path khi chạy train_agent.py.")
    # Fallback cơ bản để script không crash ngay lập tức, nhưng game sẽ không đúng
    EMPTY, WALL, FOOD, MONSTER, SIZE_WALL, YELLOW, BLUE = 0,1,2,3,30,(255,255,0),(0,0,255)
    IMAGE_PACMAN, IMAGE_GHOST = ["dummy_pac.png"], ["dummy_ghost.png"] # Tên file không quan trọng ở fallback này
    MARGIN = {"TOP":0, "LEFT":0}; BLACK, WHITE, RED = (0,0,0),(255,255,255),(255,0,0)
    SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
    class Player: 
        def __init__(self,r,c,i): self.row,self.col=r,c; self.image=None; self.rect=pygame.Rect(0,0,1,1)
    class Food: 
        def __init__(self,r,c,w,h,cl): self.row,self.col=r,c; self.image=None; self.rect=pygame.Rect(0,0,1,1)

class PacmanEnv:
    def __init__(self, map_file_path, max_steps_per_episode=500, screen_surface_for_render=None):
        self.map_file_path = map_file_path
        if not os.path.exists(self.map_file_path):
            raise FileNotFoundError(f"Không tìm thấy file map tại: {self.map_file_path}")

        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.screen_surface = screen_surface_for_render
        self.current_episode_num_for_render = 0

        # ... (các thuộc tính khác) ...
        self.N=0; self.M=0; self._map_data=[]; self.pacman=None; self.ghosts_initial_pos=[]
        self.ghosts=[]; self.foods_initial_pos=[]; self.foods=[]; self.score=0; self.initial_food_count=0
        self.ACTION_MAP = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
        self.action_space_n = len(self.ACTION_MAP)


        # --- XỬ LÝ ĐƯỜNG DẪN ẢNH - QUAN TRỌNG ---
        # Lấy đường dẫn tuyệt đối đến thư mục Source/
        # Vì pacman_env.py nằm trong Source/Environments/
        # nên thư mục cha của thư mục chứa nó chính là Source/
        current_file_dir = os.path.dirname(os.path.abspath(__file__)) # Thư mục Environments/
        source_directory = os.path.abspath(os.path.join(current_file_dir, "..")) # Thư mục Source/

        # IMAGE_PACMAN[0] từ constants.py là "images/1.png"
        # Nối nó với source_directory để tạo đường dẫn tuyệt đối
        if IMAGE_PACMAN:
            self.pacman_img_path = os.path.join(source_directory, IMAGE_PACMAN[0])
        else:
            self.pacman_img_path = os.path.join(source_directory, "images", "default_pacman.png") # Fallback

        self.ghost_imgs_paths = []
        if IMAGE_GHOST:
            for ghost_img_relative_path in IMAGE_GHOST:
                self.ghost_imgs_paths.append(os.path.join(source_directory, ghost_img_relative_path))
        else:
            self.ghost_imgs_paths.append(os.path.join(source_directory, "images", "default_ghost.png"))
        # --------------------------------------

        self._initialize_game_internals()
        self._state_dim = self._calculate_state_dim()

        if self.screen_surface:
            try: self.font = pygame.font.SysFont('Arial', 20)
            except: self.font = None
        else: self.font = None

    # ... (các hàm _read_map_file, _initialize_game_internals, _calculate_state_dim,
    #      observation_space_dim, _get_state, reset, _is_valid_position, step, render, close
    #      GIỮ NGUYÊN NHƯ PHIÊN BẢN ĐẦY ĐỦ BẠN ĐÃ CÓ Ở TRÊN) ...
    # Chỉ cần đảm bảo _initialize_game_internals sử dụng self.pacman_img_path và self.ghost_imgs_paths
    # đã được xử lý ở trên.
    def _read_map_file(self):
        pacman_start_pos = None; ghost_starts = []; food_locs = []; map_grid = []
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

        if self.screen_surface: # Cập nhật MARGIN động nếu có render
            map_pixel_width = self.M * SIZE_WALL; map_pixel_height = self.N * SIZE_WALL
            MARGIN["LEFT"] = max(0, (SCREEN_WIDTH - map_pixel_width) // 2)
            MARGIN["TOP"] = max(0, (SCREEN_HEIGHT - map_pixel_height) // 2)
        # else: MARGIN["LEFT"] = 0; MARGIN["TOP"] = 0 # Hoặc giá trị default

        # Sử dụng self.pacman_img_path đã được tạo thành đường dẫn tuyệt đối/chính xác
        self.pacman = Player(pacman_pos[0], pacman_pos[1], self.pacman_img_path)

        self.ghosts_initial_pos = list(ghost_initial_pos)
        self.ghosts = []
        for i, pos in enumerate(self.ghosts_initial_pos):
            # Sử dụng self.ghost_imgs_paths
            ghost_image_path = self.ghost_imgs_paths[i % len(self.ghost_imgs_paths)]
            self.ghosts.append(Player(pos[0], pos[1], ghost_image_path))

        self.foods_initial_pos = list(food_initial_pos)
        self.foods = []
        for r_idx in range(self.N): # Tạo food dựa trên _map_data
            for c_idx in range(self.M):
                if self._map_data[r_idx][c_idx] == FOOD:
                    self.foods.append(Food(r_idx, c_idx, SIZE_WALL // 2, SIZE_WALL // 2, YELLOW))
        self.initial_food_count = len(self.foods)
        self.score = 0; self.current_step = 0
    # ... (Các hàm còn lại không thay đổi so với phiên bản đầy đủ trước đó)
    def _calculate_state_dim(self): return self.N * self.M
    @property
    def observation_space_dim(self): return self._state_dim

    # gamestate là một mảng 1D với kích thước N*M, mỗi ô có giá trị:
    # -2: Tường (WALL)
    # -1.5: Pacman và Ghost cùng ở ô đó
    # -1: Ghost ở ô đó
    # 0.5: Food ở ô đó
    # 1: Pacman ở ô đó
    # 0: Ô trống (EMPTY)
    def _get_state(self):
        state = np.zeros(self.N * self.M, dtype=np.float32)
        for r in range(self.N):
            for c in range(self.M):
                map_1d_idx = r * self.M + c
                is_pacman_here = (self.pacman.row == r and self.pacman.col == c)
                is_ghost_here = any(ghost.row == r and ghost.col == c for ghost in self.ghosts)
                is_food_here = any(food.row == r and food.col == c for food in self.foods)
                is_wall_here = (self._map_data[r][c] == WALL)
                if is_wall_here: state[map_1d_idx] = -2.0
                elif is_pacman_here and is_ghost_here: state[map_1d_idx] = -1.5
                elif is_pacman_here: state[map_1d_idx] = 1.0
                elif is_ghost_here: state[map_1d_idx] = -1.0
                elif is_food_here: state[map_1d_idx] = 0.5
                else: state[map_1d_idx] = 0.0
        return state.flatten()
    
    def reset(self): self._initialize_game_internals(); return self._get_state()
    def _is_valid_position(self, r, c): return 0 <= r < self.N and 0 <= c < self.M and self._map_data[r][c] != WALL
    
    def step(self, action):
        self.current_step += 1; reward = -0.1; done = False
        d_row, d_col = self.ACTION_MAP[action]
        next_pac_r, next_pac_c = self.pacman.row + d_row, self.pacman.col + d_col

        if self._is_valid_position(next_pac_r, next_pac_c): self.pacman.setRC(next_pac_r, next_pac_c)
        for ghost in self.ghosts:
            valid_moves = []
            for dr_g, dc_g in self.ACTION_MAP.values():
                if self._is_valid_position(ghost.row + dr_g, ghost.col + dc_g): valid_moves.append((ghost.row + dr_g, ghost.col + dc_g))
            if valid_moves: ghost.setRC(*random.choice(valid_moves))

        for i in range(len(self.foods)-1,-1,-1):
            if self.pacman.row==self.foods[i].row and self.pacman.col==self.foods[i].col:
                self.score+=10; reward+=10.0; self.foods.pop(i); break
            
        info_status = "playing"
        for ghost in self.ghosts:
            if self.pacman.row==ghost.row and self.pacman.col==ghost.col:
                self.score-=500; reward-=500.0; done=True; info_status="lost"; break
            
        if not done:
            if not self.foods: self.score+=200; reward+=200.0; done=True; info_status="won"
            elif self.current_step >= self.max_steps_per_episode: done=True; info_status="max_steps_reached"
        
        return self._get_state(), reward, done, {"score":self.score, "steps":self.current_step, "status":info_status}
    
    def render(self, mode='text'):
        if mode == 'pygame' and self.screen_surface:
            self.screen_surface.fill(BLACK)
            map_pixel_width = self.M * SIZE_WALL; map_pixel_height = self.N * SIZE_WALL
            # Sử dụng MARGIN đã được cập nhật trong _initialize_game_internals
            margin_left_map = MARGIN["LEFT"]; margin_top_map = MARGIN["TOP"]
            for r in range(self.N):
                for c in range(self.M):
                    if self._map_data[r][c] == WALL: pygame.draw.rect(self.screen_surface, BLUE, pygame.Rect(margin_left_map + c*SIZE_WALL, margin_top_map + r*SIZE_WALL, SIZE_WALL, SIZE_WALL))
            for food_obj in self.foods:
                food_obj.rect.topleft = (margin_left_map + food_obj.col*SIZE_WALL + (SIZE_WALL-food_obj.rect.width)//2, margin_top_map + food_obj.row*SIZE_WALL + (SIZE_WALL-food_obj.rect.height)//2)
                food_obj.draw(self.screen_surface)
            for ghost_obj in self.ghosts: ghost_obj.draw(self.screen_surface) # Giả định Player.draw đã đúng
            if self.pacman: self.pacman.draw(self.screen_surface) # Giả định Player.draw đã đúng
            if self.font:
                score_text = self.font.render(f"Score: {self.score} Steps: {self.current_step}/{self.max_steps_per_episode}", True, WHITE)
                self.screen_surface.blit(score_text, (10, 10))
                if self.current_episode_num_for_render > 0:
                    ep_text = self.font.render(f"Episode: {self.current_episode_num_for_render}", True, WHITE)
                    self.screen_surface.blit(ep_text, (10, 35))
        elif mode == 'text':
            grid=[[' ']*self.M for _ in range(self.N)]
            for r_idx in range(self.N):
                for c_idx in range(self.M):
                    if self._map_data[r_idx][c_idx]==WALL: grid[r_idx][c_idx]='#'
            for f_obj in self.foods: grid[f_obj.row][f_obj.col]='.'
            for g_obj in self.ghosts: grid[g_obj.row][g_obj.col]='G'
            if self.pacman: grid[self.pacman.row][self.pacman.col]='X' if any(g.row==self.pacman.row and g.col==self.pacman.col for g in self.ghosts) else 'P'
            ep_str = f" (Ep: {self.current_episode_num_for_render})" if self.current_episode_num_for_render > 0 else ""
            print(f"--- Step: {self.current_step}, Score: {self.score}{ep_str} ---")
            for r_print_idx in range(self.N): print(" ".join(grid[r_print_idx]))
            print("-"*(self.M*2))
    def set_current_episode_num_for_render(self, ep_num: int): self.current_episode_num_for_render = ep_num
    def close(self): print("PacmanEnv: close() được gọi.")

# --- Ví dụ sử dụng để kiểm tra ---
if __name__ == '__main__':
    # ... (Phần test __main__ giữ nguyên như phiên bản đầy đủ trước đó) ...
    pass # Để ngắn gọn, bỏ qua phần test chi tiết ở đây