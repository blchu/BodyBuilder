import random

class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add_state_transition(self, s, a, r, s_prime, done):
        self.memory.append((s, a, r, s_prime, done))

        # remove oldest experience if at capacity
        while len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
