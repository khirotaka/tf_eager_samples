import tensorflow as tf
from collections import deque, namedtuple
import random
import numpy as np

Transition = namedtuple("Transition", ("state", "action", "state_next", "reward"))


class ExperienceMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        #if len(self.memory) < self.capacity:
        #    self.memory.append(None)

        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ExperienceMemory(32)


#for _ in range(32):
#    memory.push(1, 2, 3, 4)
memory.push([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
memory.push([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
memory.push([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
memory.push([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
memory.push([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
memory.push([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
memory.push([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])


def make_minibatch(BATCH_SIZE):
    transitons = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitons))
    print(batch.state)
    state_batch = tf.concat(batch.state, axis=0)
    action_batch = tf.concat(batch.action, axis=0)
    reward_batch = tf.concat(batch.reward, axis=0)
    non_final_next_states = tf.concat([s for s in batch.state_next if s is not None], axis=0)

    return batch, state_batch, action_batch, reward_batch, non_final_next_states

transitions = memory.sample(3)
#print(memory.memory)
#print(transitions)

a, b, c, d, e = make_minibatch(3)

print(a)
print(b)
print(c)
print(d)
print(e)