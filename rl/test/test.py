from rl.memory import SequentialMemory
import numpy as np

mem = SequentialMemory(limit=100000, window_length=1)

for add in range(30):
    obs = np.ones((6, 4, 20)) + add
    action = np.ones((6, 5))
    reward = add
    terminal = True

    mem.append(obs, action, reward, terminal)


batch = mem.sample(batch_size=10)
print(len(batch))
