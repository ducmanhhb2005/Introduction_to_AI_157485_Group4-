import torch
import torch.nn as nn
import torch.nn.functional as F # Thường dùng cho các hàm kích hoạt và loss
import numpy as np
import random
from collections import deque 

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1=128, hidden_dim2=128):
        """
        Khởi tạo mạng Q-Network.

        Args:
            input_dim (int): Kích thước của state vector (đầu vào mạng).
                           30
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
        if state.ndim == 1: #state đã biến thành 1 mảng 2 chiều trong đó có đúng 1 hàng
            state = state.unsqueeze(0) # Shape từ [input_dim] -> [1, input_dim]; 
        #Tại sao phải chuyển thành mảng 2D
        # Lan truyền qua các lớp, sử dụng hàm kích hoạt ReLU
        x = F.relu(self.fc1(state)) #học qua mỗi quan hệ phi tuyến tính
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x) # Lớp cuối thường không có hàm kích hoạt (cho Q-values)
        
        return q_values #tensor biểu diễn Q_value cho mỗi hành động

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
       
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): # lấy ra đông thời cảnh báo cái nào đáng wanring vì ko có next states
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
        #Chân thực về cái nào là True or False(None) trong next_states. Sử dụng để biết sau này còn tính Q_value tránh những cái ko hợp lệ ra về next
        return state_batch, action_batch, reward_batch, non_final_next_states_tensor, done_batch, non_final_mask

    def __len__(self):
        """
        Trả về số lượng kinh nghiệm hiện có trong buffer.
        """
        return len(self.buffer)


