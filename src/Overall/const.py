ALGORITHM: str = "MINIMAX"

LEVEL_TO_ALGORITHM = {
    "LEVEL1": "BFS",
    "LEVEL2": "BFS",
    "LEVEL3": "Local Search",
    "LEVEL4": "Minimax"
}
ALGORITHMS = ["BFS", "DFS", "Local Search", "Minimax", "AlphaBeta", "Expectimax"]
# DEFINE COLOR
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# DEFINE MAP
SIZE_WALL: int = 30   #Kích thước 1 ô
DEFINE_WIDTH: int = 6   #Độ dày của viền tường
BLOCK_SIZE: int = SIZE_WALL // 2 #Lấy phần nguyên   #Kích thước 1 viên thức ăn
NUM_GHOSTS_FOR_STATE = 4
# Entity
EMPTY = 0
WALL = 1
FOOD = 2
MONSTER = 3

# Setup screen
WIDTH: int = 1200
HEIGHT: int = 600
FPS: int = 300

MARGIN = { #lề trên và lề trái để hiển thị bản đồ game
    "TOP": 0,
    "LEFT": 0
}


# IMAGE
IMAGE_GHOST = ["images/Blinky.png", "images/Pinky.png", "images/Inky.png", "images/Clyde.png"]
IMAGE_PACMAN = ["images/1.png", "images/2.png", "images/3.png", "images/4.png"]
