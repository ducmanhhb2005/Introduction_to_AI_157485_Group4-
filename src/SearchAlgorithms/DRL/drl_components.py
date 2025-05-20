import torch
import torch.nn as nn
import torch.nn.functional as F # Thường dùng cho các hàm kích hoạt và loss
import numpy as np
import random
from collections import deque # deque là một cấu trúc dữ liệu hiệu quả cho buffer

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1=128, hidden_dim2=128):
        """
        Khởi tạo mạng Q-Network.

        Args:
            input_dim (int): Kích thước của state vector (đầu vào mạng).
                             Ví dụ: N * M nếu state là flattened map.
            output_dim (int): Số lượng hành động khả thi (đầu ra mạng).
                              Ví dụ: 4 cho Pac-Man (Lên, Xuống, Trái, Phải).
            hidden_dim1 (int): Số units trong lớp ẩn thứ nhất.
            hidden_dim2 (int): Số units trong lớp ẩn thứ hai.
        """
        super(DQNNetwork, self).__init__() # Gọi hàm __init__ của lớp cha (nn.Module)

        # Định nghĩa các lớp (layers) của mạng
        # Một mạng MLP (Multi-Layer Perceptron) đơn giản với 2 lớp ẩn
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # Lớp fully connected thứ nhất
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2) # Lớp fully connected thứ hai
        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # Lớp đầu ra

    def forward(self, state):
        """
        Định nghĩa quá trình lan truyền tiến (forward pass) của mạng.
        Input: state (torch.Tensor) - tensor biểu diễn trạng thái.
        Output: q_values (torch.Tensor) - tensor chứa Q-value cho mỗi hành động.
        """
        # Đảm bảo state là tensor và có kiểu float, và có chiều batch nếu cần
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        # Nếu state là vector 1D (ví dụ, khi chọn action cho 1 state), thêm chiều batch
        if state.ndim == 1:
            state = state.unsqueeze(0) # Shape từ [input_dim] -> [1, input_dim]

        # Lan truyền qua các lớp, sử dụng hàm kích hoạt ReLU
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x) # Lớp cuối thường không có hàm kích hoạt (cho Q-values)

        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Khởi tạo Replay Buffer.

        Args:
            capacity (int): Số lượng kinh nghiệm (transitions) tối đa có thể lưu trữ.
        """
        # deque (double-ended queue) rất hiệu quả cho việc thêm và xóa ở cả hai đầu.
        # Khi buffer đầy, việc thêm mới sẽ tự động xóa phần tử cũ nhất.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Lưu một kinh nghiệm (transition) vào buffer.

        Args:
            state (np.ndarray): Trạng thái hiện tại.
            action (int): Hành động đã thực hiện.
            reward (float): Phần thưởng nhận được.
            next_state (np.ndarray or None): Trạng thái kế tiếp. Có thể là None nếu done=True.
            done (bool): True nếu episode kết thúc sau hành động này.
        """
        # Không cần np.expand_dims ở đây nếu state/next_state đã là vector 1D
        # Việc chuyển đổi sang tensor sẽ được thực hiện khi sample
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Lấy ngẫu nhiên một batch các kinh nghiệm từ buffer.

        Args:
            batch_size (int): Số lượng kinh nghiệm cần lấy.

        Returns:
            tuple: Một tuple chứa các batch tensor:
                   (state_batch, action_batch, reward_batch,
                    non_final_next_states_tensor, done_batch, non_final_mask)
        """
        if len(self.buffer) < batch_size:
            # Nếu không đủ kinh nghiệm, có thể trả về None hoặc raise Exception
            # Trong thực tế, hàm learn() của agent sẽ kiểm tra điều này trước
            return None

        # Lấy ngẫu nhiên `batch_size` transitions từ buffer
        transitions = random.sample(self.buffer, batch_size)

        # "Giải nén" các transitions thành các list riêng biệt
        # states, actions, rewards, next_states, dones sẽ là các tuple
        # Ví dụ: states = (state1, state2, ..., state_batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Chuyển đổi sang PyTorch tensors
        # np.array() được dùng để chuyển tuple các numpy array (states, next_states)
        # thành một numpy array 2D trước khi tạo tensor.
        state_batch = torch.tensor(np.array(states), dtype=torch.float32)
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1) # Shape: [batch_size, 1]
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) # Shape: [batch_size, 1]
        done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)    # Shape: [batch_size, 1]

        # Xử lý `next_states` cẩn thận vì một số có thể là `None` (khi `done` là True)
        # Tạo một list chỉ chứa các `next_state` không phải `None`
        non_final_next_states_list = [s_next for s_next in next_states if s_next is not None]
        
        # Tạo tensor cho các non-final next states
        if len(non_final_next_states_list) > 0:
            non_final_next_states_tensor = torch.tensor(np.array(non_final_next_states_list), dtype=torch.float32)
        else:
            # Nếu tất cả next_states trong batch đều là None (tất cả done=True),
            # tạo một tensor rỗng với số chiều đúng để tránh lỗi khi tính toán sau này.
            # Shape của state là (state_dim,)
            state_dim = state_batch.shape[1] if state_batch.ndim > 1 and state_batch.shape[0] > 0 else 0
            if state_dim == 0 and len(states) > 0 and states[0] is not None: # Xử lý trường hợp state là 1D
                state_dim = states[0].shape[0]
            non_final_next_states_tensor = torch.empty((0, state_dim), dtype=torch.float32)


        # Tạo một mask (mặt nạ) boolean để chỉ ra những `next_state` nào là non-final (không phải None)
        # Mask này sẽ hữu ích khi tính Q-target cho các non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)

        return state_batch, action_batch, reward_batch, non_final_next_states_tensor, done_batch, non_final_mask

    def __len__(self):
        """
        Trả về số lượng kinh nghiệm hiện có trong buffer.
        """
        return len(self.buffer)

# --- Ví dụ sử dụng (để kiểm tra các component này) ---
if __name__ == '__main__':
    # Kiểm tra DQNNetwork
    print("--- Kiểm tra DQNNetwork ---")
    input_dim_test = 10  # Giả sử state có 10 chiều
    output_dim_test = 4  # Giả sử có 4 hành động
    network = DQNNetwork(input_dim_test, output_dim_test)
    print("Cấu trúc mạng:", network)

    # Tạo một state giả để test
    dummy_state_np = np.random.rand(input_dim_test).astype(np.float32)
    dummy_state_tensor = torch.tensor(dummy_state_np, dtype=torch.float32)

    # Test forward pass với 1 state
    q_values_single = network(dummy_state_tensor) # input [input_dim_test]
    print("Q-values cho 1 state (shape input [10]):", q_values_single)
    print("Shape output:", q_values_single.shape) # Mong đợi: [1, output_dim_test]

    # Test forward pass với batch state
    dummy_batch_state_np = np.random.rand(5, input_dim_test).astype(np.float32) # batch_size = 5
    dummy_batch_state_tensor = torch.tensor(dummy_batch_state_np, dtype=torch.float32)
    q_values_batch = network(dummy_batch_state_tensor) # input [5, input_dim_test]
    print("\nQ-values cho batch 5 states (shape input [5,10]):", q_values_batch)
    print("Shape output:", q_values_batch.shape) # Mong đợi: [5, output_dim_test]


    # Kiểm tra ReplayBuffer
    print("\n--- Kiểm tra ReplayBuffer ---")
    buffer_capacity_test = 100
    replay_buffer = ReplayBuffer(buffer_capacity_test)

    # Push một vài kinh nghiệm giả
    for i in range(10):
        s = np.random.rand(input_dim_test).astype(np.float32)
        a = random.randint(0, output_dim_test - 1)
        r = random.random()
        s_next_done = random.random() > 0.8 # 20% là done
        s_next = np.random.rand(input_dim_test).astype(np.float32) if not s_next_done else None
        d = s_next_done
        replay_buffer.push(s, a, r, s_next, d)
        print(f"Pushed: s_shape={s.shape}, a={a}, r={r:.2f}, s_next_type={type(s_next)}, d={d}")

    print(f"\nĐộ dài buffer hiện tại: {len(replay_buffer)}")

    # Sample một batch
    batch_size_test = 5
    if len(replay_buffer) >= batch_size_test:
        print(f"\nSampling batch_size={batch_size_test}...")
        sample_result = replay_buffer.sample(batch_size_test)
        if sample_result:
            s_batch, a_batch, r_batch, nf_next_s_batch, d_batch, nf_mask = sample_result
            print("Shape state_batch:", s_batch.shape)         # Mong đợi: [batch_size_test, input_dim_test]
            print("Shape action_batch:", a_batch.shape)       # Mong đợi: [batch_size_test, 1]
            print("Shape reward_batch:", r_batch.shape)       # Mong đợi: [batch_size_test, 1]
            print("Shape non_final_next_states_batch:", nf_next_s_batch.shape) # Mong đợi: [số_lượng_non_final, input_dim_test]
            print("Shape done_batch:", d_batch.shape)         # Mong đợi: [batch_size_test, 1]
            print("Shape non_final_mask:", nf_mask.shape)     # Mong đợi: [batch_size_test]
            print("non_final_mask:", nf_mask)
            print("Số lượng non_final_next_states thực tế:", nf_next_s_batch.size(0))
            print("Số lượng True trong non_final_mask:", torch.sum(nf_mask).item())

        else:
            print("Không thể sample do buffer quá nhỏ.")
    else:
        print("Buffer quá nhỏ để sample.")

    # Kiểm tra trường hợp tất cả next_states đều là None
    print("\n--- Kiểm tra ReplayBuffer với tất cả next_states là None ---")
    replay_buffer_all_done = ReplayBuffer(10)
    for _ in range(5):
        s = np.random.rand(input_dim_test).astype(np.float32)
        a = random.randint(0, output_dim_test - 1)
        r = random.random()
        replay_buffer_all_done.push(s, a, r, None, True) # next_state là None, done là True

    if len(replay_buffer_all_done) >= 3:
        s_b, a_b, r_b, nf_ns_b, d_b, nf_m = replay_buffer_all_done.sample(3)
        print("Shape non_final_next_states_batch (all done):", nf_ns_b.shape) # Mong đợi [0, input_dim_test]
        print("non_final_mask (all done):", nf_m) # Mong đợi [False, False, False]