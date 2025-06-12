# Source/Algorithms/DRL/evaluate_agent.py
import os
import sys
import pygame
import numpy as np 
import matplotlib.pyplot as plt 


# Lấy đường dẫn đến thư mục chứa file script này (DRL))
_current_script_directory = os.path.dirname(os.path.abspath(__file__))

# Đi lên các cấp thư mục để đến thư mục Source/ và thư mục gốc của project

_algorithms_directory = os.path.abspath(os.path.join(_current_script_directory, "..")) 
_source_directory = os.path.abspath(os.path.join(_algorithms_directory, ".."))      
_project_root_directory = os.path.abspath(os.path.join(_source_directory, ".."))     

# Thêm các thư mục cần thiết vào sys.path để Python có thể tìm thấy các module
if _source_directory not in sys.path:
    sys.path.insert(0, _source_directory) # Ưu tiên thư mục src
if _project_root_directory not in sys.path: #Input/ hoặc các tài nguyên khác nằm ở gốc project
    sys.path.insert(0, _project_root_directory)
# ------------------------

# Bây giờ có thể import các module từ bên trong Source/
try:
    from Environments.pacman_env import PacmanEnv
   
    from dqn_agent import DQNAgent
   
    from Overall.const import WIDTH, HEIGHT 
except ImportError as e:
    print(f"Lỗi import trong evaluate_agent.py: {e}")
    print("Vui lòng kiểm tra lại cấu trúc thư mục và phần xử lý sys.path.")
    print(f"sys.path hiện tại: {sys.path}")
    exit()


# Cấu hình Đánh giá 
# Đường dẫn đến model đã huấn luyện
MODEL_FILENAME = "dqn_pacman_ghosts_final.pth" 
# Đường dẫn tương đối đến thư mục lưu model, TÍNH TỪ THƯ MỤC GỐC CỦA PROJECT
MODEL_SAVE_SUBDIR_FROM_PROJECT_ROOT = os.path.join("src", "SearchAlgorithmsxDRL", "DRL", "saved_models_rendered") #Map 11x20
#MODEL_SAVE_SUBDIR_FROM_PROJECT_ROOT = os.path.join("src", "SearchAlgorithmsxDRL", "DRL", "saved_models_pacman_with_ghosts")
MODEL_SAVE_DIR_ABSOLUTE = os.path.join(_project_root_directory, MODEL_SAVE_SUBDIR_FROM_PROJECT_ROOT)
# Không cần os.makedirs ở đây nữa vì thư mục này phải tồn tại do train_agent.py tạo ra.
MODEL_TO_EVALUATE_PATH = os.path.join(MODEL_SAVE_DIR_ABSOLUTE, MODEL_FILENAME)


# Map để đánh giá (tính từ thư mục gốc của project)
MAP_FOR_EVALUATION_RELATIVE_FROM_PROJECT_ROOT = os.path.join("InputMap",  "map11x20.txt") #Tùy chỉnh map
#MAP_FOR_EVALUATION_RELATIVE_FROM_PROJECT_ROOT = os.path.join("InputMap",  "map20x22.txt") #Tùy chỉnh map
MAP_FOR_EVALUATION_ABSOLUTE = os.path.join(_project_root_directory, MAP_FOR_EVALUATION_RELATIVE_FROM_PROJECT_ROOT)

NUM_EVALUATION_EPISODES = 10
RENDER_EVALUATION_CONFIG = True  # Render or not
FPS_FOR_EVALUATION = 10      # Tốc độ khung hình khi render
EVALUATION_EPSILON = 0.01    # Epsilon rất nhỏ để agent chủ yếu khai thác
# ---------------------------

def evaluate_trained_agent():
    print(f"--- Bắt đầu Đánh giá Agent ---")
    print(f"Model đang đánh giá: {MODEL_TO_EVALUATE_PATH}")
    print(f"Map sử dụng: {MAP_FOR_EVALUATION_ABSOLUTE}")

    pygame.init() # Khởi tạo tất cả các module của Pygame
    screen = None
    clock = None

    # Sử dụng một biến cục bộ để quyết định có render hay không trong lần chạy này
    # và khởi tạo nó từ biến cấu hình toàn cục.
    should_render_this_run = RENDER_EVALUATION_CONFIG

    if should_render_this_run:
        try:
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("DRL Pacman Evaluation")
            clock = pygame.time.Clock()
            print("Màn hình Pygame đã được khởi tạo để render đánh giá.")
        except pygame.error as e:
            print(f"LỖI PYGAME khi khởi tạo màn hình render: {e}. Sẽ đánh giá mà không render.")
            should_render_this_run = False # Thay đổi biến cục bộ nếu có lỗi
    elif not pygame.display.get_init(): # Vẫn init display tối thiểu nếu không render
        try:
            pygame.display.init() # Chỉ init module display là đủ cho việc load ảnh
            print("Thông báo: Pygame display đã được khởi tạo tối thiểu (không render).")
        except pygame.error as e:
            print(f"Cảnh báo: Không thể init pygame.display tối thiểu: {e}")


    # Kiểm tra file map
    if not os.path.exists(MAP_FOR_EVALUATION_ABSOLUTE):
        print(f"LỖI: File map để đánh giá không tồn tại tại '{MAP_FOR_EVALUATION_ABSOLUTE}'.")
        # Tạo file map ví dụ nếu nó không tồn tại (giống logic trong train_agent.py)
        print(f"Đang tạo file map ví dụ tại: {MAP_FOR_EVALUATION_ABSOLUTE}")
        try:
            os.makedirs(os.path.dirname(MAP_FOR_EVALUATION_ABSOLUTE), exist_ok=True)
            with open(MAP_FOR_EVALUATION_ABSOLUTE, "w") as f:
                f.write("7 7\n"); f.write("1 1 1 1 1 1 1\n"); f.write("1 2 0 0 0 3 1\n"); f.write("1 0 1 0 1 0 1\n")
                f.write("1 0 0 2 0 0 1\n"); f.write("1 3 1 0 1 0 1\n"); f.write("1 2 0 0 0 2 1\n"); f.write("1 1 1 1 1 1 1\n"); f.write("3 3\n")
            print("Đã tạo file map ví dụ.")
        except Exception as create_map_e:
            print(f"Lỗi khi tạo file map ví dụ: {create_map_e}")
            if pygame.get_init(): pygame.quit()
            return

    env = PacmanEnv(map_file_path=MAP_FOR_EVALUATION_ABSOLUTE,
                    screen_surface_for_render=screen if should_render_this_run else None)

    state_dim = env.observation_space_dim
    action_dim = env.action_space_n

    if state_dim == 0:
        print("LỖI: state_dim từ PacmanEnv là 0. Kiểm tra lại PacmanEnv, đặc biệt là _calculate_state_dim và việc đọc map.")
        if pygame.get_init(): pygame.quit()
        return

    # Khởi tạo agent với các siêu tham số mặc định của DQNAgent,
    # vì các siêu tham số huấn luyện không quan trọng khi chỉ load model để đánh giá.pyth
    agent = DQNAgent(state_dim, action_dim)

    if not os.path.exists(MODEL_TO_EVALUATE_PATH):
        print(f"Lỗi: Không tìm thấy file model tại '{MODEL_TO_EVALUATE_PATH}'")
        if pygame.get_init(): pygame.quit()
        return

    try:
        agent.load_model(MODEL_TO_EVALUATE_PATH,continue_training=False)
    except Exception as e:
        print(f"Lỗi khi tải model '{MODEL_TO_EVALUATE_PATH}': {e}")
        if pygame.get_init(): pygame.quit()
        return

    agent.epsilon = EVALUATION_EPSILON  # Đặt epsilon rất thấp
    agent.policy_net.eval() #  Chuyển policy_net sang evaluation mode
    if hasattr(agent, 'target_net') and agent.target_net is not None: # Kiểm tra target_net tồn tại
        agent.target_net.eval() # Chuyển cả target_net sang evaluation mode


    print(f"\nBắt đầu chạy {NUM_EVALUATION_EPISODES} episodes đánh giá (Epsilon={agent.epsilon:.2f})...")
    all_episode_rewards = []
    all_episode_steps = []
    all_episode_status = []

    running_script = True
    for episode in range(1, NUM_EVALUATION_EPISODES + 1):
        if not running_script: break

        state = env.reset()
        if should_render_this_run:
            env.set_current_episode_num_for_render(episode)

        episode_reward = 0
        episode_step_count = 0
        done = False
        info = {} # Khởi tạo info để tránh lỗi nếu vòng lặp while không chạy

        # Giới hạn số bước trong env.max_steps_per_episode
        # hoặc cho đến khi done (thắng/thua)
        for current_step_in_episode in range(1, 2000 + 1):
            episode_step_count = current_step_in_episode # Cập nhật số bước thực tế

            if should_render_this_run and screen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Đánh giá bị dừng bởi người dùng.")
                        running_script = False; break
                if not running_script: break
            
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action) # env.step sẽ tự xử lý max_steps
            
            state = next_state
            episode_reward += reward

            if should_render_this_run and screen:
                env.render(mode='pygame')
                pygame.display.flip()
                if clock: clock.tick(FPS_FOR_EVALUATION)
            
            if done: # Kiểm tra cờ done từ env.step()
                break
        
        if not running_script: break # Thoát sớm nếu người dùng đóng cửa sổ

        all_episode_rewards.append(episode_reward)
        all_episode_steps.append(episode_step_count)
        all_episode_status.append(info.get("status", "unknown"))

        print(f"Đ.giá Ep: {episode}/{NUM_EVALUATION_EPISODES} | "
              f"Steps: {episode_step_count} | "
              f"Total Reward: {episode_reward:.2f} | "
              f"Status: {info.get('status', 'N/A')}")

    print("\n--- Kết Quả Đánh Giá ---")
    if all_episode_rewards:
        print(f"Tổng số episodes đã chạy: {len(all_episode_rewards)}")
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        print(f"Reward trung bình: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Reward cao nhất: {np.max(all_episode_rewards):.2f}")
        print(f"Reward thấp nhất: {np.min(all_episode_rewards):.2f}")
        print(f"Số bước trung bình/episode: {np.mean(all_episode_steps):.2f}")
        
        status_counts = {status: all_episode_status.count(status) for status in set(all_episode_status)}
        print("Thống kê trạng thái kết thúc episodes:")
        for status, count in status_counts.items():
            print(f"  - {status}: {count} lần")
    else:
        print("Không có dữ liệu episode nào được chạy hoàn chỉnh để thống kê.")

    # Vẽ đồ thị
    if all_episode_rewards :
        try:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(all_episode_rewards,
                     bins=max(1, min(10, len(set(all_episode_rewards)))),
                     edgecolor='black', color='skyblue')
            plt.title('Phân phối Tổng Reward (Episodes Đánh giá)')
            plt.xlabel('Tổng Reward'); plt.ylabel('Số lượng Episodes'); plt.grid(True, linestyle='--', alpha=0.7)

            plt.subplot(1, 2, 2)
            plt.plot(all_episode_rewards, marker='o', linestyle='-', color='coral', label='Reward/Episode')
            if len(all_episode_rewards) >= 10:
                moving_avg = np.convolve(all_episode_rewards, np.ones(10)/10, mode='valid')
                plt.plot(np.arange(len(moving_avg)) + (10//2 -1) , moving_avg, color='dodgerblue', linewidth=2, label='Moving Avg (10 ep)')
            plt.title('Tổng Reward theo từng Episode (Đánh giá)'); plt.xlabel('Episode'); plt.ylabel('Tổng Reward'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            eval_plot_save_path = os.path.join(MODEL_SAVE_DIR_ABSOLUTE, "evaluation_plots.png")
            plt.savefig(eval_plot_save_path)
            print(f"Đồ thị kết quả đánh giá đã lưu tại: {eval_plot_save_path}")

            if should_render_this_run: # Chỉ hiển thị nếu có render và không có lỗi trước đó
                print("Đang cố gắng hiển thị đồ thị...")
                plt.show()
                print("Cửa sổ đồ thị đã được đóng.")

        except Exception as e:
            print(f"Lỗi khi vẽ hoặc hiển thị đồ thị Matplotlib: {e}")
    elif not ('plt' in sys.modules and 'matplotlib' in sys.modules):
        print("Matplotlib chưa được import hoặc có lỗi khi import. Bỏ qua việc vẽ đồ thị.")


if __name__ == '__main__':
    try:
        evaluate_trained_agent()
    except Exception as e:
        print(f"LỖI KHÔNG MONG MUỐN TRONG QUÁ TRÌNH ĐÁNH GIÁ: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pygame.get_init():
            pygame.quit()
            print("Thông báo (finally): Đã gọi pygame.quit().")