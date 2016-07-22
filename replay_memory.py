import numpy as np
import random

class ReplayMemory():

    def __init__(self, capacity, state_dims):
        self.capacity = capacity
        
        self.states = np.empty([self.capcity, state_dims[0], state_dims[1]])
        self.actions = np.empty([self.capacity])
        self.rewards = np.empty([self.capacity])
        self.terminals = np.empty([self.capacity])

        self.end_pointer = 0
        self.filled = False

    def add_state_transition(self, s, a, r, done):
        self.states[self.end_pointer] = s
        self.actions[self.end_pointer] = a
        self.rewards[self.end_pointer] = r
        self.terminals[self.end_pointer] = done

        # advance pointer and set filled if loops back to start
        self.end_pointer = (self.end_pointer+1) % self.capacity
        if self.end_pointer == 0:
            self.filled = True


    def sample(self, batch_size):
        current_size = self.capacity if filled else self.end_pointer-1
        assert current_size > batch_size
        indices = random.sample(current_size, batch_size)

        # end pointer index edge case
        if self.end_pointer in indices:
            end_val = self.end_pointer
            end_index = indices.index(end_val)
            while end_val in indices:
                end_val = random.randint(0, current_size)
            indices[end_index] = end_val

        batch_states = self.states(indices)
        batch_actions = self.actions(indices)
        batch_rewards = self.rewards(indices)
        batch_terminals = self.terminals(indices)
        batch_newstates = self.states([i+1 % self.capacity for i in indices])

        return (batch_states, batch_actions, batch_rewards, batch_terminals, batch_newstates)
