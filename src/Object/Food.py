# Source/Object/Food.py
import pygame

# Giả sử constants.py nằm trong thư mục Source/,
# và Food.py nằm trong Source/Object/
try:
    from Overall.constants import WHITE, SIZE_WALL, MARGIN, YELLOW # Thêm YELLOW nếu chưa có
except ImportError:
    print("Cảnh báo (Food.py): Không thể import constants.py. Sử dụng giá trị mặc định.")
    WHITE = (255, 255, 255)
    SIZE_WALL = 30
    MARGIN = {"TOP": 0, "LEFT": 0}
    YELLOW = (255,255,0)


class Food:
    def __init__(self, row: int, col: int, width: int, height: int, color=YELLOW): # Thêm color mặc định
        self.row = row
        self.col = col
        self.width = width # Lưu lại để có thể dùng sau nếu cần
        self.height = height # Lưu lại
        self.color = color

        self.image = pygame.Surface([width, height])
        self.image.fill(WHITE) # Nên fill màu nền trước khi set colorkey
        self.image.set_colorkey(WHITE) # Màu trắng sẽ trở nên trong suốt
        pygame.draw.ellipse(self.image, color, [0, 0, width, height])

        self.rect = self.image.get_rect()
        # Vị trí ban đầu của rect.topleft sẽ được tính toán dựa trên row, col.
        # PacmanEnv.render() sẽ chịu trách nhiệm đặt lại vị trí này
        # cho chính xác trên màn hình render, bao gồm cả việc căn giữa food trong ô.
        self._update_rect_position()


    def _update_rect_position(self):
        """
        Cập nhật vị trí top-left của self.rect để food nằm giữa ô logic.
        PacmanEnv có thể ghi đè các giá trị này khi render để căn giữa toàn bộ map.
        """
        # Căn giữa food trong ô SIZE_WALL x SIZE_WALL
        cell_top = self.row * SIZE_WALL + MARGIN["TOP"]
        cell_left = self.col * SIZE_WALL + MARGIN["LEFT"]
        
        self.rect.left = cell_left + (SIZE_WALL - self.width) // 2
        self.rect.top = cell_top + (SIZE_WALL - self.height) // 2

    def draw(self, screen: pygame.Surface):
        """Vẽ Food lên màn hình."""
        # Hàm draw sẽ sử dụng self.rect.topleft đã được PacmanEnv.render()
        # điều chỉnh (nếu cần) hoặc vị trí từ _update_rect_position()
        screen.blit(self.image, self.rect.topleft)

    def getRC(self) -> list[int, int]:
        """Lấy vị trí (hàng, cột) hiện tại."""
        return [self.row, self.col]

    # setRC có thể không cần thiết cho Food vì Food thường không di chuyển
    # Nếu cần, bạn có thể thêm:
    # def setRC(self, row: int, col: int):
    #     self.row = row
    #     self.col = col
    #     self._update_rect_position()

# --- Ví dụ sử dụng cơ bản (để kiểm tra cú pháp) ---
if __name__ == '__main__':
    try:
        pygame.init()
        screen_test = None
        try:
            screen_test = pygame.display.set_mode((100, 100))
            print("Pygame display initialized for Food.py test.")
        except pygame.error as e:
            print(f"Không thể set_mode cho test Food.py: {e}.")

        # Test tạo Food
        food1 = Food(row=1, col=1, width=SIZE_WALL // 2, height=SIZE_WALL // 2, color=YELLOW)
        print(f"Food 1 rect: {food1.rect}") # Vị trí sẽ dựa trên MARGIN và SIZE_WALL

        if screen_test:
            screen_test.fill((50,50,50)) # Nền xám để thấy food
            # Để vẽ đúng, bạn cần tính toán vị trí dựa trên margin của màn hình test
            # Hoặc đơn giản là vẽ tại vị trí rect đã tính
            food1.draw(screen_test)
            pygame.display.flip()
            # pygame.time.wait(1000)

    except ImportError:
        print("Không thể import Pygame hoặc constants. Bỏ qua phần test Food.py.")
    except Exception as e:
        print(f"Lỗi không xác định trong phần test Food.py: {e}")
    finally:
        pygame.quit()