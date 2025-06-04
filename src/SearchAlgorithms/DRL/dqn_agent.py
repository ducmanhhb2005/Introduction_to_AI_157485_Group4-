import torch
import torch.optim as optim # Chứa các thuật toán tối ưu như Adam, SGD
import torch.nn.functional as F # Cho hàm loss (ví dụ: smooth_l1_loss)
import random # Để chọn hành động ngẫu nhiên (epsilon-greedy)
import numpy as np # Mặc dù PyTorch xử lý tensor, numpy vẫn hữu ích cho một số thao tác
import os 
from typing import Union
# Import các component đã tạo từ file drl_components.py
# Điều chỉnh đường dẫn import nếu cần, dựa trên cấu trúc thư mục của bạn
try:
    from .drl_components import DQNNetwork, ReplayBuffer
except ImportError: # Fallback nếu chạy file này độc lập (ít khả thi cho agent)
    from drl_components import DQNNetwork, ReplayBuffer


class DQNAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-4, # Tốc độ học (thường nhỏ hơn cho DRL)
                 gamma: float = 0.99,         # Hệ số chiết khấu (discount factor)
                 epsilon_start: float = 1.0,  # Giá trị epsilon ban đầu cho khám phá
                 epsilon_end: float = 0.01,   # Giá trị epsilon cuối cùng (vẫn còn một chút khám phá)
                 epsilon_decay_rate: float = 0.995, # Tốc độ giảm epsilon sau mỗi episode/step
                 buffer_capacity: int = 10000, # Sức chứa của Replay Buffer
                 batch_size: int = 64,         # Số lượng mẫu lấy từ buffer để học mỗi lần
                 target_update_freq: int = 100): # Tần suất cập nhật Target Network (tính bằng số lần gọi hàm learn())
        """
        Khởi tạo DQN Agent.

        Args:
            state_dim (int): Kích thước của state vector.
            action_dim (int): Số lượng hành động khả thi.
            learning_rate (float): Tốc độ học cho optimizer.
            gamma (float): Hệ số chiết khấu cho phần thưởng tương lai.
            epsilon_start (float): Epsilon ban đầu.
            epsilon_end (float): Epsilon tối thiểu.
            epsilon_decay_rate (float): Hệ số giảm epsilon.
            buffer_capacity (int): Kích thước tối đa của Replay Buffer.
            batch_size (int): Kích thước batch khi lấy mẫu từ buffer.
            target_update_freq (int): Số lần gọi `learn()` trước khi cập nhật Target Network.
        """
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end # Đổi tên để rõ ràng hơn
        self.epsilon_decay = epsilon_decay_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Xác định thiết bị (GPU nếu có, nếu không thì CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent đang sử dụng thiết bị: {self.device}")

        # Khởi tạo Policy Network và Target Network
        # Cả hai mạng đều có cùng kiến trúc
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)

        # Sao chép trọng số từ policy_net sang target_net ban đầu
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Đặt target_net ở chế độ evaluation (không tính gradient, không cập nhật trọng số trực tiếp)
        self.target_net.eval()

        # Khởi tạo Optimizer (ví dụ: Adam) cho policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Khởi tạo Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Biến đếm số lần đã gọi hàm learn() để cập nhật target_net
        self.learn_step_counter = 0

    def select_action(self, state: np.ndarray) -> int:
        """
        Chọn hành động dựa trên state hiện tại theo chiến lược epsilon-greedy.

        Args:
            state (np.ndarray): Vector trạng thái hiện tại.

        Returns:
            int: Hành động được chọn (một số nguyên).
        """
        # Khám phá (Exploration)
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim) # Chọn hành động ngẫu nhiên
        # Khai thác (Exploitation)
        else:
            with torch.no_grad(): # Không cần tính gradient khi chỉ dự đoán (inference)
                # Chuyển state (numpy array) sang tensor PyTorch và đưa lên device
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Đưa state qua policy_net để lấy Q-values
                q_values = self.policy_net(state_tensor)
                # Chọn hành động có Q-value cao nhất
                action = q_values.max(1)[1].item() # .max(1) trả về (giá trị max, index max) dọc theo chiều 1
                                                  # [1] để lấy index, .item() để lấy giá trị Python từ tensor 0-dim
        return action

    def learn(self) -> Union[float, None]:
        """
        Cập nhật trọng số của Policy Network dựa trên một batch kinh nghiệm từ Replay Buffer.

        Returns:
            float or None: Giá trị loss nếu việc học diễn ra, None nếu buffer chưa đủ.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None # Chưa đủ kinh nghiệm trong buffer để học

        # Lấy một batch ngẫu nhiên các transitions từ replay buffer
        sample_result = self.replay_buffer.sample(self.batch_size)
        if sample_result is None: # Phòng trường hợp sample trả về None (dù đã kiểm tra len)
            return None
        
        states_batch, actions_batch, rewards_batch, \
        non_final_next_states_batch, dones_batch, non_final_mask = sample_result

        # Chuyển tất cả các tensor trong batch lên device (CPU/GPU)
        states_batch = states_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        rewards_batch = rewards_batch.to(self.device)
        if non_final_next_states_batch.shape[0] > 0: # Chỉ chuyển nếu có non_final_next_states
            non_final_next_states_batch = non_final_next_states_batch.to(self.device)
        dones_batch = dones_batch.to(self.device)
        non_final_mask = non_final_mask.to(self.device)


        # 1. Tính Q(s_t, a_t) hiện tại từ Policy Network
        #    Chúng ta cần Q-values cho các hành động thực sự đã được thực hiện (actions_batch)
        #    policy_net(states_batch) trả về Q-values cho TẤT CẢ hành động từ mỗi state trong batch.
        #    .gather(1, actions_batch) chọn ra Q-value tương ứng với hành động đã thực hiện.
        current_q_values = self.policy_net(states_batch).gather(1, actions_batch)

        # 2. Tính giá trị V(s_{t+1}) cho các next_states từ Target Network
        #    V(s_{t+1}) = max_{a'} Q_target(s_{t+1}, a')
        #    Đối với các terminal states (done=True), giá trị này là 0.
        next_q_values_target = torch.zeros(self.batch_size, 1, device=self.device) # Khởi tạo bằng 0 cho tất cả
        
        # Chỉ tính Q-values cho các non-final next states (những state mà non_final_mask là True)
        if non_final_next_states_batch.shape[0] > 0: # Đảm bảo có ít nhất một non-final next state
            with torch.no_grad(): # Không cần gradient cho target network khi tính target
                next_q_values_target[non_final_mask] = self.target_net(non_final_next_states_batch).max(1)[0].unsqueeze(1)
                # .max(1)[0] lấy giá trị Q-value lớn nhất
                # .unsqueeze(1) để đảm bảo shape là [num_non_final, 1]
                # .detach() (không thực sự cần vì đã có torch.no_grad(), nhưng để rõ ràng)

        # 3. Tính Q-value mục tiêu (Expected Q-value / Bellman target)
        #    y_j = r_j                            nếu s'_j là terminal
        #    y_j = r_j + gamma * max_a' Q_target(s'_j, a') nếu s'_j không phải terminal
        #    (1 - dones_batch) sẽ làm cho phần gamma * next_q_values_target bằng 0 nếu done=True
        expected_q_values = rewards_batch + (self.gamma * next_q_values_target * (1 - dones_batch))

        # 4. Tính hàm mất mát (Loss Function)
        #    So sánh current_q_values với expected_q_values
        #    Smooth L1 Loss (Huber loss) thường ổn định hơn MSELoss cho DQN
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        # Hoặc bạn có thể dùng MSE:
        # loss = F.mse_loss(current_q_values, expected_q_values)

        # 5. Tối ưu hóa Policy Network
        self.optimizer.zero_grad()  # Xóa các gradient từ bước trước
        loss.backward()             # Tính toán gradient của loss theo các tham số của policy_net
        
        # (Tùy chọn) Cắt gradient (Gradient Clipping) để tránh gradient quá lớn làm mất ổn định
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Giới hạn giá trị gradient
        # Hoặc torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # Giới hạn norm của gradient

        self.optimizer.step()       # Cập nhật trọng số của policy_net

        # Tăng biến đếm số lần học
        self.learn_step_counter += 1

        # 6. Cập nhật Target Network định kỳ
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print(f"Bước học {self.learn_step_counter}: Đã cập nhật Target Network.")
        
        return loss.item() # Trả về giá trị loss (dưới dạng số Python) để theo dõi

    def decay_epsilon(self):
        """Giảm giá trị epsilon sau mỗi episode hoặc một số bước nhất định."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min: # Đảm bảo epsilon không nhỏ hơn epsilon_min
                self.epsilon = self.epsilon_min
        # Hoặc một cách khác để decay (linear decay):
        # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_value)

    def save_model(self, file_path: str):
        """Lưu trọng số của Policy Network."""
        try:
            torch.save(self.policy_net.state_dict(), file_path)
            print(f"Model đã được lưu vào: {file_path}")
        except Exception as e:
            print(f"Lỗi khi lưu model: {e}")

    def load_model(self, file_path: str):
        """Tải trọng số cho Policy Network và đồng bộ Target Network."""
        try:
            self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Đồng bộ target net
            self.policy_net.eval() # Chuyển sang chế độ đánh giá
            self.target_net.eval()
            print(f"Model đã được tải từ: {file_path}")
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file model tại {file_path}. Bỏ qua việc tải model.")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")

# --- Ví dụ sử dụng cơ bản (để kiểm tra cú pháp, không phải để huấn luyện thực tế) ---
if __name__ == '__main__':
    print("--- Kiểm tra DQNAgent ---")
    dummy_state_dim = 10
    dummy_action_dim = 4
    agent = DQNAgent(dummy_state_dim, dummy_action_dim, buffer_capacity=100, batch_size=10, target_update_freq=5)

    print(f"Epsilon ban đầu: {agent.epsilon}")

    # Tạo một vài kinh nghiệm giả để đẩy vào buffer
    for _ in range(20): # Tạo nhiều hơn batch_size
        s = np.random.rand(dummy_state_dim).astype(np.float32)
        a = agent.select_action(s) # Dùng select_action để có cả khám phá
        r = random.uniform(-1, 1)
        s_next_done = random.random() > 0.8
        s_next = np.random.rand(dummy_state_dim).astype(np.float32) if not s_next_done else None
        d = s_next_done
        agent.replay_buffer.push(s, a, r, s_next, d)

    print(f"Độ dài buffer: {len(agent.replay_buffer)}")

    # Thử gọi hàm learn vài lần
    if len(agent.replay_buffer) >= agent.batch_size:
        for i in range(10): # Gọi learn 10 lần
            loss_val = agent.learn()
            if loss_val is not None:
                print(f"Lần học {i+1}, Loss: {loss_val:.4f}, Epsilon sau learn: {agent.epsilon:.3f}")
            else:
                print(f"Lần học {i+1}: Buffer chưa đủ, không học.")
            # Giảm epsilon thủ công để test (trong training loop thực tế sẽ gọi agent.decay_epsilon())
            if agent.epsilon > agent.epsilon_min: agent.epsilon *= agent.epsilon_decay

    # Test save/load
    test_model_path = "dummy_dqn_agent.pth"
    agent.save_model(test_model_path)
    
    # Tạo agent mới và load
    new_agent = DQNAgent(dummy_state_dim, dummy_action_dim)
    new_agent.load_model(test_model_path)
    print("Load model thành công vào agent mới.")

    # Xóa file model test
    if os.path.exists(test_model_path):
        os.remove(test_model_path)