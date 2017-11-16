import random
import numpy as np
from resNet.buildNetwork import buildNetwork
from collections import deque



class DeepQNetwork(object):

    def __init__(self,
                 n_actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 epsilon_greedy=0.9,
                 memory_size=500,
                 batch_size=32,
                 input_shape=(480, 320)):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_greedy = epsilon_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque()
        self.model = buildNetwork(input_shape, n_actions)

    # choose an action based on q value with epsilon greedy
    def choose_action(self, observation):

        if np.random.uniform() < self.epsilon_greedy:  # choose max q
            predicted_q_val = self.model.predict_on_batch(observation)
            action = np.argmax(predicted_q_val)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    # store the transitions for training
    def store_transition(self, s, a, r, s_, done):
        if len(self.memory) > self.n_memory:
            self.memory.popleft()

        self.memory.append([s, a, r, s_, done])

    def create_batch(self):
        sample = random.choice(self.memory, self.batch_size)
        sample = np.asarray(sample)
        s = sample[:, 0]
        a = sample[:, 1].astype(np.int)
        r = sample[:, 2]
        s_ = sample[:, 3]
        done = sample[:, 4].astype(np.int8)

        X_batch = np.vstack(s)
        y_batch = self.model.predict_on_batch(X_batch)
        y_batch[np.arrange(self.batch_size), a] = \
            r + self.gamma*np.max(self.model.predict_on_batch(np.vstack(s_)))*(1-done)

        return X_batch, y_batch

    # train the neural network
    def learn(self):
        if len(self.memory) > self.batch_size:
            X_batch, y_batch = self.creat_batch()
            self.model.train_on_batch(X_batch, y_batch)







