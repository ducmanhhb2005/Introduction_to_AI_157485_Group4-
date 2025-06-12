# dqn_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from typing import Optional 

try:
    from .drl_components import DQNNetwork, ReplayBuffer
except ImportError:
    from drl_components import DQNNetwork, ReplayBuffer

class DQNAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_rate: float = 0.995,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent đang sử dụng thiết bị: {self.device}")

        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval() # Target net nên ở eval() mode khi tính target,
                               # nhưng policy_net sẽ được đặt lại ở train() khi bắt đầu huấn luyện mới

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.learn_step_counter = 0

 #Chon hành động
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        return action

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        sample_result = self.replay_buffer.sample(self.batch_size)
        if sample_result is None:
            print("Cảnh báo (DQNAgent.learn): sample_result từ ReplayBuffer là None sau khi đã kiểm tra len.")
            return None
        try:
            states_batch, actions_batch, rewards_batch, \
            non_final_next_states_batch, dones_batch, non_final_mask = sample_result
        except ValueError as e:
            print(f"LỖI UNPACKING trong DQNAgent.learn: {e}")
            print(f"DEBUG: sample_result có {len(sample_result) if isinstance(sample_result, tuple) else 'không phải tuple'} phần tử.")
            print(f"DEBUG: sample_result = {sample_result}")
            return None

        states_batch = states_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        rewards_batch = rewards_batch.to(self.device)
        if non_final_next_states_batch.shape[0] > 0:
            non_final_next_states_batch = non_final_next_states_batch.to(self.device)
        dones_batch = dones_batch.to(self.device)
        non_final_mask = non_final_mask.to(self.device)

        current_q_values = self.policy_net(states_batch).gather(1, actions_batch)
        next_q_values_target = torch.zeros(self.batch_size, 1, device=self.device)
        if non_final_next_states_batch.shape[0] > 0:
            with torch.no_grad():
                next_q_values_target[non_final_mask] = self.target_net(non_final_next_states_batch).max(1)[0].unsqueeze(1)
        
        expected_q_values = rewards_batch + (self.gamma * next_q_values_target * (1 - dones_batch))
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward() #Tính toán gradient
        self.optimizer.step() #tHay đổi trọng số của mạng dựa vào gradient
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save_model(self, file_path: str):
        """Lưu trọng số của Policy Network, trạng thái Optimizer và Epsilon."""
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'learn_step_counter': self.learn_step_counter
            }, file_path)
            print(f"Model, Optimizer và Epsilon đã được lưu vào: {file_path}")
        except Exception as e:
            print(f"Lỗi khi lưu model: {e}")

    def load_model(self, file_path: str, continue_training: bool = True): # Thêm cờ continue_training
        """
        Tải trọng số cho Policy Network, Target Network, trạng thái Optimizer và Epsilon.
        Args:
            file_path (str): Đường dẫn đến file model.
            continue_training (bool): Đặt True nế muốn tiếp tục huấn luyện (policy_net sẽ ở .train()).
                                      Đặt False nếu chỉ muốn đánh giá (policy_net sẽ ở .eval()).
        """
        try:
            checkpoint = torch.load(file_path, map_location=self.device) 
            
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            
            # Tải target_net state dict nếu có, nếu không thì đồng bộ từ policy_net
            if 'target_net_state_dict' in checkpoint:
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            else: 
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Lấy epsilon và learn_step_counter, có giá trị mặc định nếu không tìm thấy
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min) # Nếu không có, đặt là epsilon_min
            self.learn_step_counter = checkpoint.get('learn_step_counter', 0)

            if continue_training:
                self.policy_net.train() # Đặt lại về chế độ huấn luyện nếu muốn tiếp tục train
                print(f"Model, Optimizer và Epsilon đã được tải từ: {file_path} để tiếp tục huấn luyện.")
                print(f"Epsilon được khôi phục: {self.epsilon:.4f}")
            else:
                self.policy_net.eval() # Chuyển sang chế độ đánh giá
                print(f"Model đã được tải từ: {file_path} ở chế độ đánh giá.")
            
            self.target_net.eval() # Target net luôn ở eval mode khi tính toán target Q-values

        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file model tại {file_path}. Bỏ qua việc tải model.")
        except KeyError as e:
            print(f"Lỗi KeyError khi tải model: {e}. File model có thể không chứa tất cả các key cần thiết .")
            print("Đang thử tải chỉ policy_net state_dict (nếu file model là phiên bản cũ).")
            try: # Thử tải chỉ trọng số policy net nếu checkpoint cũ
                self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                if continue_training: self.policy_net.train()
                else: self.policy_net.eval()
                self.target_net.eval()
                print("Cảnh báo: Chỉ tải được policy_net_state_dict. Optimizer và Epsilon sẽ bắt đầu lại.")
            except Exception as fallback_e:
                print(f"Lỗi khi thử tải chỉ policy_net_state_dict: {fallback_e}")
        except Exception as e:
            print(f"Lỗi không xác định khi tải model: {e}")