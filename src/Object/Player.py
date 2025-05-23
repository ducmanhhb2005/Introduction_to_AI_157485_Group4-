# Source/Object/Player.py
import pygame
import os

try:
    from Overall.constants import SIZE_WALL, MARGIN
except ImportError:
    print("Cảnh báo (Player.py): Không thể import constants.py. Sử dụng giá trị mặc định.")
    SIZE_WALL = 30
    MARGIN = {"TOP": 0, "LEFT": 0} # MARGIN này có thể không còn quá quan trọng nếu Env quản lý vị trí vẽ cuối cùng

class Player:
    def __init__(self, row: int, col: int, initial_fileImage: str):
        self.row = row
        self.col = col
        self.current_fileImage = initial_fileImage

        self._load_and_transform_image(initial_fileImage, rotate_angle=0)

        # self.rect sẽ được tạo bên trong _load_and_transform_image hoặc được cập nhật sau đó
        # Khởi tạo một rect cơ bản ban đầu, nó sẽ được đặt đúng vị trí bởi _update_rect_position
        if hasattr(self, 'image') and self.image: # Kiểm tra self.image tồn tại
             self.rect = self.image.get_rect()
        else: # Fallback nếu image không load được
            self.rect = pygame.Rect(0,0, SIZE_WALL, SIZE_WALL)
        self._update_rect_position()


    def _load_and_transform_image(self, file_image_path: str, rotate_angle: float = 0):
        try:
            loaded_image = pygame.image.load(file_image_path)
            processed_image = loaded_image.convert_alpha()
        except pygame.error as e:
            print(f"LỖI PYGAME khi tải hoặc convert ảnh (Player): {e} cho file '{file_image_path}'")
            self.image = pygame.Surface([SIZE_WALL, SIZE_WALL]) # Gán trực tiếp cho self.image
            self.image.fill((255, 0, 0))
            return

        try:
            scaled_image = pygame.transform.scale(processed_image, (SIZE_WALL, SIZE_WALL))
        except pygame.error as e:
            print(f"LỖI PYGAME khi scale ảnh (Player): {e} cho file '{file_image_path}'")
            self.image = pygame.Surface([SIZE_WALL, SIZE_WALL]) # Gán trực tiếp cho self.image
            self.image.fill((255, 100, 0))
            return

        if rotate_angle != 0:
            try:
                # Quan trọng: xoay ảnh gốc đã scale, không xoay self.image nhiều lần
                self.image = pygame.transform.rotate(scaled_image, rotate_angle)
            except pygame.error as e:
                print(f"LỖI PYGAME khi rotate ảnh (Player): {e} cho file '{file_image_path}'")
                self.image = scaled_image # Sử dụng ảnh đã scale nếu rotate lỗi
        else:
            self.image = scaled_image

    def _update_rect_position(self):
        """
        Cập nhật vị trí top-left của self.rect dựa trên self.row, self.col,
        SIZE_WALL, và MARGIN từ constants.
        PacmanEnv có thể ghi đè các giá trị này khi render để căn giữa.
        """
        if hasattr(self, 'image') and self.image: # Đảm bảo image tồn tại
            # Lấy rect mới từ image hiện tại để đảm bảo kích thước đúng sau khi xoay
            current_center = self.rect.center if hasattr(self, 'rect') else (0,0) # Lưu tâm cũ nếu có
            self.rect = self.image.get_rect()
            # Đặt vị trí dựa trên logic grid
            self.rect.top = self.row * SIZE_WALL + MARGIN["TOP"]
            self.rect.left = self.col * SIZE_WALL + MARGIN["LEFT"]
            # Nếu bạn muốn giữ tâm sau khi xoay (phức tạp hơn và có thể không cần thiết nếu Env căn chỉnh lại):
            # self.rect.center = current_center
            # self.rect.top = self.row * SIZE_WALL + MARGIN["TOP"] # Sau đó lại căn theo grid có thể làm lệch tâm
        else: # Fallback nếu image chưa được tạo
            self.rect = pygame.Rect(
                self.col * SIZE_WALL + MARGIN["LEFT"],
                self.row * SIZE_WALL + MARGIN["TOP"],
                SIZE_WALL,
                SIZE_WALL
            )


    def change_state(self, rotate: float, fileImage: str):
        self.current_fileImage = fileImage
        self._load_and_transform_image(fileImage, rotate_angle=rotate)
        # Sau khi image thay đổi (và có thể cả kích thước do xoay),
        # rect cần được cập nhật và đặt lại vị trí.
        self._update_rect_position() # Hàm này sẽ lấy rect mới từ self.image và đặt vị trí

    def draw(self, screen: pygame.Surface):
        # Hàm draw sẽ sử dụng self.rect đã được PacmanEnv.render() điều chỉnh vị trí (nếu cần)
        # hoặc vị trí tính toán từ _update_rect_position()
        screen.blit(self.image, self.rect.topleft)

    def getRC(self) -> list[int, int]:
        return [self.row, self.col]

    def setRC(self, row: int, col: int):
        self.row = row
        self.col = col
        self._update_rect_position() # Luôn cập nhật vị trí pixel khi row/col thay đổi

    def move(self, d_R: int, d_C: int):
        self.rect.top += d_R
        self.rect.left += d_C

    def touch_New_RC(self, row: int, col: int) -> bool:
        target_top = row * SIZE_WALL + MARGIN["TOP"]
        target_left = col * SIZE_WALL + MARGIN["LEFT"]
        # So sánh topleft của rect hiện tại với topleft mục tiêu
        return self.rect.top == target_top and self.rect.left == target_left

# --- Ví dụ sử dụng cơ bản (để kiểm tra cú pháp) ---
if __name__ == '__main__':
    # ... (Phần test __main__ có thể giữ nguyên hoặc điều chỉnh tương tự như trước) ...
    # ... (Quan trọng là phải có pygame.init() và pygame.display.set_mode() để test đầy đủ)
    pass # Bỏ qua phần test chi tiết ở đây để tập trung vào class