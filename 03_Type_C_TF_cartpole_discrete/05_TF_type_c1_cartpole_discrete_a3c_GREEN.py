import os
import sys
import gym
import pylab
import numpy as np
import time
import tensorflow as tf
import multiprocessing
import threading
# N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = 6

env_name = "CartPole-v1"
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance
# env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

MAX_EP_STEP = 500
UPDATE_GLOBAL_ITER = 10
discount_factor = 0.9  # reward discount
ENTROPY_BETA = 0.001

actor_lr = 0.001    # learning rate for actor
critic_lr = 0.001    # learning rate for critic

episode = 0

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

# Network for the Actor Critic
class A3CAgent(object):
    def __init__(self, sess, scope, master_agent = None):
        self.sess = sess
        # get size of state and action
        self.action_size = action_size
        self.state_size = state_size
        self.value_size = 1
        
        self.hidden1, self.hidden2 = 128, 128
        
        self.master_agent = master_agent
        self.scope = scope

        # create model for actor and critic network
        with tf.variable_scope(self.scope):
            self._init_input()
            self.build_model()
            self._init_op()

    def _init_input(self):
        # with tf.variable_scope('input'):
        self.state = tf.placeholder(tf.float32,  [None, self.state_size], name='state')
        self.action = tf.placeholder(tf.int32,   [None, ],               name='action')
        self.q_target = tf.placeholder(tf.float32, [None, 1],          name='q_target')

    # make loss function for Policy Gradient
    # [log(action probability) * discounted_rewards] will be input for the back prop
    # we add entropy of action probability to loss
    def _init_op(self):
        if self.scope == 'master':   # get global network
            with tf.name_scope(self.scope):
                self.actor_optimizer = tf.train.RMSPropOptimizer(actor_lr)
                self.critic_optimizer = tf.train.RMSPropOptimizer(critic_lr)
        else:   # local net, calculate losses
            with tf.variable_scope(self.scope):
                # with tf.variable_scope('td_error'):
                self.td_error = tf.subtract(self.q_target, self.value, name='td_error')

                # with tf.variable_scope('critic_loss'):
                self.critic_loss = tf.reduce_mean(tf.square(self.td_error))

                # with tf.variable_scope('actor_loss'):
                log_prob = tf.reduce_sum(tf.log(self.policy + 1e-5) * tf.one_hot(self.action, self.action_size, dtype=tf.float32), axis=1, keep_dims=True)
                exp_v = log_prob * tf.stop_gradient(self.td_error)
                entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-5),
                                         axis=1, keep_dims=True)  # encourage exploration
                self.exp_v = ENTROPY_BETA * entropy + exp_v
                self.actor_loss = tf.reduce_mean(-self.exp_v)

                # with tf.name_scope('local_gradients'):
                self.actor_gradients = tf.gradients(self.actor_loss, self.actor_params) #calculate gradients for the network weights
                self.critic_gradients = tf.gradients(self.critic_loss, self.critic_params)

            # with tf.name_scope('sync'): # update local and global network weights
            # with tf.name_scope('pull'):
            zipped_actor_vars = zip(self.actor_params, self.master_agent.actor_params)
            zipped_critic_vars = zip(self.critic_params, self.master_agent.critic_params)
            self.pull_actor_params_op = [l_a_p.assign(g_a_p) for l_a_p, g_a_p in zipped_actor_vars]
            self.pull_critic_params_op = [l_c_p.assign(g_c_p) for l_c_p, g_c_p in zipped_critic_vars]

            # with tf.name_scope('push'):
            zipped_actor_vars = zip(self.actor_gradients, self.master_agent.actor_params)
            zipped_critic_vars = zip(self.critic_gradients, self.master_agent.critic_params)
            self.update_actor_op = self.master_agent.actor_optimizer.apply_gradients(zipped_actor_vars)
            self.update_critic_op = self.master_agent.critic_optimizer.apply_gradients(zipped_critic_vars)

    # neural network structure of the actor and critic
    def build_model(self):

        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(0.1)

        with tf.variable_scope("actor"):

            actor_hidden = tf.layers.dense(self.state, self.hidden1, tf.nn.tanh, kernel_initializer=w_init,
                                        bias_initializer=b_init)

            self.actor_predict = tf.layers.dense(actor_hidden, self.action_size, kernel_initializer=w_init,
                                                   bias_initializer=b_init)

            self.policy = tf.nn.softmax(self.actor_predict)
    
        with tf.variable_scope("critic"):

            critic_hidden = tf.layers.dense(inputs=self.state, units = self.hidden1, activation=tf.nn.tanh,  # tanh activation
                kernel_initializer=w_init, bias_initializer=b_init, name='fc1_c')

            critic_predict = tf.layers.dense(inputs=critic_hidden, units = self.value_size, activation=None,
                kernel_initializer=w_init, bias_initializer=b_init, name='fc2_c')
            self.value = critic_predict

        self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_actor_op, self.update_critic_op], feed_dict)

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_actor_params_op, self.pull_critic_params_op])

    # get action from policy network
    def get_action(self, state):
        """
            Choose action based on observation

            Arguments:
                state: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """ 
        # Reshape observation to (num_features, 1)
        state_t =  state[np.newaxis, :]
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.policy, feed_dict={self.state: state_t})
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
        
# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, env, sess, name, COORD, master_agent):

        self.sess = sess
        self.env = env
        self.coordinator = COORD
        self.name = name
        self.agent = A3CAgent(sess, name, master_agent)
        self.master_agent = master_agent

        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        self.buffer_q_target = []
        self.train_steps = 1

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def append_sample(self, state, action, reward):
        self.buffer_state.append(state)
        self.buffer_action.append(action)
        self.buffer_reward.append(reward)

    # update policy network and value network every episode
    def train_model(self, next_state, done):
        if done:
            value_next_state = 0   # terminal
        else:
            value_next_state = self.sess.run(self.agent.value, {self.agent.state: next_state[np.newaxis, :]})[0][0]

        for reward in self.buffer_reward[::-1]:    # reverse buffer r
            value_next_state = reward + discount_factor * value_next_state
            self.buffer_q_target.append(value_next_state)

        self.buffer_q_target.reverse()

        feed_dict={
            self.agent.state: np.vstack(self.buffer_state),
            self.agent.action: np.array(self.buffer_action),
            self.agent.q_target: np.vstack(self.buffer_q_target)
        } 
        
        self.agent.update_global(feed_dict) # actual training step, update global A3CAgent
        self.agent.pull_global() # get global parameters to local A3CAgent    
        
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        self.buffer_q_target = []

    def work(self):
        global episode
        train_steps = 0
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        
        scores, episodes = [], []
        avg_score = 0

        start_time = time.time()
        
        while time.time() - start_time < 5*60 and avg_score < 495:
            
            done = False
            score = 0
            state = self.env.reset()

            while not done and score < MAX_EP_STEP:
                # every time step we do train from the replay memory
                score += 1
                
                # if self.name == 'W_0':
                #     self.env.render()
                
                train_steps += 1
                # get action for the current state and go one step in environment
                action = self.agent.get_action(state)
                
                # make step in environment
                next_state, reward, done, _ = self.env.step(action) 
                
                # save the sample <state, action, reward> to the memory
                self.append_sample(state, action, reward)
                
                if train_steps % 10 == 0 or done:   # update global and assign to local net
                    self.train_model(next_state, done)
                    
                # swap observation
                state = next_state
                
                # train when epsisode finished
                if done or score == MAX_EP_STEP:
                    episode += 1
                    # self.train_model(next_state, done)
                    
                    # every episode, plot the play time
                    scores.append(score)
                    episodes.append(episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])
                    
                    print("episode :{:5d}".format(episode), "/ score :{:5d}".format(score))
                    
                    break

        e = int(time.time() - start_time)
        print('Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

if __name__ == "__main__":
    sess = tf.Session()

    global_agent = A3CAgent(sess, "master")  # we only need its params
    workers = []
    
    COORD = tf.train.Coordinator()
    
    # Create workers
    for index in range(N_WORKERS):
        env = gym.make(env_name).unwrapped
        i_name = 'W_%i' % index   # worker name
        workers.append(Worker(env, sess, i_name, COORD, global_agent))

    sess.run(tf.global_variables_initializer())
    
    worker_threads = []
    for worker in workers: #start workers
        job = worker.work
        thread = threading.Thread(target = job)
        thread.start()
        worker_threads.append(thread)
        
    COORD.join(worker_threads)  # wait for termination of workers
    
    tf.summary.FileWriter(graph_path + "/", sess.graph)
    saver = tf.train.Saver()
    saver.save(sess, model_path+ "/")
    

