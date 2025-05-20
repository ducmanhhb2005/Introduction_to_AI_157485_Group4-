# train_agent.py
import sys
import os
import matplotlib.pyplot as plt # Để vẽ đồ thị (tùy chọn, cài đặt: pip install matplotlib)
import numpy as np # Thường hữu ích
import pygame # Import Pygame

# --- Xử lý PYTHONPATH để import các module từ thư mục cha ---
# Lấy đường dẫn đến thư mục chứa file train_agent.py này (DRL/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Đi lên 2 cấp để đến thư mục Source/ (DRL -> Algorithms -> Source)
source_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# Thêm thư mục Source vào sys.path nếu nó chưa có
if source_dir not in sys.path:
    sys.path.insert(0, source_dir) # Thêm vào đầu để ưu tiên

# (Tùy chọn) Nếu thư mục Input nằm ngoài Source và ngang cấp với Source
# (tức là trong thư mục gốc của project), bạn cũng có thể thêm thư mục gốc của project.
project_root_dir = os.path.abspath(os.path.join(source_dir, "..")) # Từ Source đi lên 1 cấp
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# ------------------------------------------------------------

# Bây giờ có thể import các module
try:
    from Environments.pacman_env import PacmanEnv
    from SearchAlgorithms.DRL.dqn_agent import DQNAgent
    from Overall.const import WIDTH, HEIGHT # Import kích thước màn hình từ constants.py
except ImportError as e:
    print(f"Lỗi import trong train_agent.py: {e}")
    print(f"Kiểm tra lại cấu trúc thư mục và sys.path. sys.path hiện tại: {sys.path}")
    exit()

# --- Cấu hình Huấn luyện ---
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 300
# Đường dẫn file map (điều chỉnh nếu cần)
# Giả sử file train_agent.py nằm trong Source/Algorithms/DRL/
# Và Input/ nằm trong thư mục gốc của project (YourPacmanProject/Input/)
MAP_FILE_RELATIVE_TO_PROJECT_ROOT = "Input/Level1/map_for_drl.txt" # Tạo file map riêng cho DRL nếu muốn
# MAP_FILE_RELATIVE_TO_PROJECT_ROOT = "Input/Level1/map1.txt" # Hoặc dùng map1.txt
MAP_FILE = os.path.join(project_root_dir, MAP_FILE_RELATIVE_TO_PROJECT_ROOT)


# Siêu tham số cho DQNAgent
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_RATE = 0.995 # Giảm 0.5% epsilon sau mỗi episode
BUFFER_CAPACITY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100

# Lưu trữ model
SAVE_MODEL_EVERY_EPISODES = 100
MODEL_SAVE_DIR = os.path.join(current_dir, "saved_models_rendered") # Thư mục lưu model
MODEL_FILENAME_PREFIX = "dqn_pacman_agent_rendered"
LOAD_PRETRAINED_MODEL = False
PRETRAINED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_FILENAME_PREFIX}_final.pth")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Tạo thư mục nếu chưa có

# --- Cấu hình cho việc RENDER đồ họa ---
RENDER_TRAINING_CONFIG = True  # Đặt True để hiển thị game, False để huấn luyện ngầm (nhanh hơn)
FPS_WHILE_RENDERING = 10 # FPS khi hiển thị game (chậm hơn để dễ quan sát)
# -----------------------------------------

def train():
    print(f"Bắt đầu quá trình huấn luyện...")
    print(f"Sử dụng file map: {MAP_FILE}")

    # --- KHỞI TẠO PYGAME VÀ MÀN HÌNH ---
    pygame.init() # Khởi tạo tất cả các module pygame
    screen = None
    clock = None

    should_render_this_run = RENDER_TRAINING_CONFIG # Bắt đầu với giá trị cấu hình toàn cục

    if should_render_this_run:
        try:
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("DRL Pacman Training - Rendering")
            clock = pygame.time.Clock()
            print(f"Thông báo: Màn hình Pygame ({WIDTH}x{HEIGHT}) đã được khởi tạo để render.")
        except pygame.error as e:
            print(f"LỖI PYGAME khi khởi tạo màn hình render: {e}. Sẽ huấn luyện mà không render.")
            should_render_this_run = False # Thay đổi biến cục bộ, không ảnh hưởng đến RENDER_TRAINING_CONFIG toàn cục
    elif not pygame.display.get_init():
        try:
            pygame.display.init()
            print("Thông báo: Pygame display đã được khởi tạo tối thiểu (không render).")
        except pygame.error as e:
            print(f"Cảnh báo: Không thể init pygame.display tối thiểu: {e}")

    # -----------------------------------------------------------------

    if not os.path.exists(MAP_FILE):
        print(f"LỖI: Không tìm thấy file map tại '{MAP_FILE}'.")
        # Tạo file map ví dụ nếu nó không tồn tại
        print(f"Đang tạo file map ví dụ tại: {MAP_FILE}")
        os.makedirs(os.path.dirname(MAP_FILE), exist_ok=True)
        with open(MAP_FILE, "w") as f:
            f.write("7 7\n")
            f.write("1 1 1 1 1 1 1\n")
            f.write("1 2 0 0 0 3 1\n")
            f.write("1 0 1 0 1 0 1\n")
            f.write("1 0 0 2 0 0 1\n")
            f.write("1 3 1 0 1 0 1\n")
            f.write("1 2 0 0 0 2 1\n")
            f.write("1 1 1 1 1 1 1\n")
            f.write("3 3\n")
        print("Đã tạo file map ví dụ. Vui lòng kiểm tra và chạy lại nếu cần.")
        # return # Có thể thoát ở đây hoặc để nó thử chạy với map vừa tạo

    # Truyền screen vào PacmanEnv nếu có render
    env = PacmanEnv(map_file_path=MAP_FILE,
                    max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                    screen_surface_for_render=screen if should_render_this_run else None) # Sử dụng biến cục bộ
    state_dim = env.observation_space_dim
    action_dim = env.action_space_n

    if state_dim == 0:
        print("LỖI: state_dim từ PacmanEnv là 0. Kiểm tra lại PacmanEnv, đặc biệt là _calculate_state_dim và kích thước map.")
        return

    print(f"Kích thước State (Observation Space Dim): {state_dim}")
    print(f"Số lượng Hành động (Action Space Size): {action_dim}")

    agent = DQNAgent(state_dim, action_dim,
                     learning_rate=LEARNING_RATE, gamma=GAMMA,
                     epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                     epsilon_decay_rate=EPSILON_DECAY_RATE,
                     buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE,
                     target_update_freq=TARGET_UPDATE_FREQ)

    if LOAD_PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Đang tải model đã huấn luyện từ: {PRETRAINED_MODEL_PATH}")
        agent.load_model(PRETRAINED_MODEL_PATH)
        agent.epsilon = EPSILON_END # Bắt đầu với epsilon thấp nếu tải model
    else:
        print("Bắt đầu huấn luyện từ đầu (không tải model trước).")

    episode_rewards = []
    episode_avg_losses = []
    episode_steps_taken = [] # Đổi tên từ episode_steps để rõ ràng hơn

    print(f"\n--- Bắt đầu vòng lặp huấn luyện cho {NUM_EPISODES} episodes ---")
    running_script = True
    for episode in range(1, NUM_EPISODES + 1):
        if not running_script: break

        state = env.reset()
        if should_render_this_run: # Truyền số episode vào env để render nếu có
            env.set_current_episode_num_for_render(episode)

        current_episode_reward = 0
        current_episode_total_loss = 0
        num_learn_calls_this_episode = 0
        steps_this_episode = 0


        for step_num in range(1, MAX_STEPS_PER_EPISODE + 1):
            steps_this_episode = step_num # Lưu lại số bước thực tế

            if should_render_this_run and screen: # Sử dụng biến cục bộ
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Người dùng đã đóng cửa sổ. Kết thúc huấn luyện sớm.")
                        running_script = False
                        break
                if not running_script: break

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state if not done else None, done)
            state = next_state
            current_episode_reward += reward
            loss = agent.learn()
            if loss is not None:
                current_episode_total_loss += loss
                num_learn_calls_this_episode += 1

            if should_render_this_run and screen:
                env.render(mode='pygame')
                pygame.display.flip()
                if clock: clock.tick(FPS_WHILE_RENDERING)

            if done:
                break
        
        if not running_script: break

        agent.decay_epsilon()
        episode_rewards.append(current_episode_reward)
        avg_loss = current_episode_total_loss / num_learn_calls_this_episode if num_learn_calls_this_episode > 0 else 0
        episode_avg_losses.append(avg_loss)
        episode_steps_taken.append(steps_this_episode)

        print(f"Episode: {episode}/{NUM_EPISODES} | "
              f"Steps: {steps_this_episode}/{MAX_STEPS_PER_EPISODE} | "
              f"Total Reward: {current_episode_reward:.2f} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Status: {info.get('status', 'N/A')}")

        if episode % SAVE_MODEL_EVERY_EPISODES == 0:
            save_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_FILENAME_PREFIX}_episode_{episode}.pth")
            agent.save_model(save_path)

    final_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_FILENAME_PREFIX}_final.pth")
    agent.save_model(final_model_path)
    print(f"\n--- Huấn luyện hoàn tất ---")
    print(f"Model cuối cùng đã được lưu tại: {final_model_path}")

    # Vẽ đồ thị
    try:
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards, label='Total Reward')
        if len(episode_rewards) >= 10:
            moving_avg_rewards = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
            plt.plot(np.arange(len(moving_avg_rewards)) + (10//2 -1) , moving_avg_rewards, label='Moving Avg (10 ep)', color='red')
        plt.title('Total Reward per Episode'); plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.legend(); plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(episode_avg_losses, label='Avg Loss')
        if len(episode_avg_losses) >= 10:
            moving_avg_losses = np.convolve(episode_avg_losses, np.ones(10)/10, mode='valid')
            plt.plot(np.arange(len(moving_avg_losses)) + (10//2 -1), moving_avg_losses, label='Moving Avg (10 ep)', color='red')
        plt.title('Average Loss per Episode'); plt.xlabel('Episode'); plt.ylabel('Average Loss'); plt.legend(); plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(episode_steps_taken, label='Steps Taken')
        if len(episode_steps_taken) >= 10:
            moving_avg_steps = np.convolve(episode_steps_taken, np.ones(10)/10, mode='valid')
            plt.plot(np.arange(len(moving_avg_steps)) + (10//2 -1), moving_avg_steps, label='Moving Avg (10 ep)', color='red')
        plt.title('Steps per Episode'); plt.xlabel('Episode'); plt.ylabel('Number of Steps'); plt.legend(); plt.grid(True)

        plt.tight_layout()
        plot_save_path = os.path.join(MODEL_SAVE_DIR, "training_performance_plots.png")
        plt.savefig(plot_save_path)
        print(f"Đồ thị kết quả huấn luyện đã được lưu tại: {plot_save_path}")
        if should_render_this_run: # Chỉ show plot nếu không phải đang chạy ngầm hoàn toàn
             plt.show()
    except ImportError:
        print("Matplotlib chưa được cài đặt. Bỏ qua việc vẽ đồ thị. (pip install matplotlib)")
    except Exception as e:
        print(f"Lỗi khi vẽ đồ thị: {e}")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"Lỗi không mong muốn trong quá trình huấn luyện: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        print("Thông báo: Đã gọi pygame.quit().")