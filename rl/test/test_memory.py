from rl.memory import SequentialMemory
import numpy as np

mem = SequentialMemory(limit=100, window_length=1)

def get_state(ndim=(2,)):
    i = 0
    while True:
        i += 1
        yield np.ones(ndim) * i

env = get_state()

for idx, s in enumerate(env):
    a, r, t = 1, 1, 0
    if idx % 3 == 0:
        t = 1
    mem.append(s, a, r, t)
    print(s)

    if idx >= 99:
        break


for x in mem.observations:
    print(x)

mem.sample(batch_size=4, batch_idxs=[1, 2, 3, 4])

