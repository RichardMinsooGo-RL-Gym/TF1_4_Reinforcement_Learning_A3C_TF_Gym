import gym
import tensorflow as tf
import numpy as np
import time, datetime

import sys
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()

import multiprocessing
import threading
# N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = 6

env_name = "Acrobot-v1"
env = gym.make(env_name)
# env.seed(1)     # reproducible, general Policy gradient has high variance
# np.random.seed(123)
# tf.set_random_seed(456)  # reproducible
env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

training_time = 5*60
ep_trial_step = 10000

UPDATE_GLOBAL_ITER = 10
discount_factor = 0.9  # reward discount
# ENTROPY_BETA = 0.001

learning_rate = 0.005

episode = 0
step = 0

game_name =  sys.argv[0][:-3]

model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

# Network for the Actor Critic
class A3CAgent(object):
    def __init__(self, scope, master_agent = None):
        
        # if you want to see Cartpole learning, then change to True
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        
        # these is hyper parameters for the ActorCritic
        self.hidden1, self.hidden2 = 64, 64
        
        self.master_agent = master_agent
        self.scope = scope
        
        if scope == "master":   # get global network
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32,  [None, self.state_size], name='state')
                # parameters of actor and critic net
                # self.model_params, self.critic_params = self.build_model(scope)[-1:]
                self.policy, self.value, self.model_params = self.build_model(scope)
                                
        else:   # local net, calculate losses
            with tf.variable_scope(self.scope):
                self.state = tf.placeholder(tf.float32,  [None, self.state_size], name='state')
                self.action = tf.placeholder(tf.float32, [None, self.action_size], name="action")
                self.q_target = tf.placeholder(tf.float32, [None, 1],          name='q_target')

                self.policy, self.value, self.model_params = self.build_model(scope)
                
                # with tf.variable_scope('td_error'):
                # A_t = R_t - V(S_t)
                # self.td_error = tf.subtract(self.q_target, self.value, name='td_error')
                self.td_error = self.q_target - self.value

                # with tf.variable_scope('actor_loss'):
                # Policy loss
                self.log_p = self.action * tf.log(tf.clip_by_value(self.policy,1e-10,1.))
                self.log_lik = self.log_p * tf.stop_gradient(self.td_error)
                self.actor_loss = -tf.reduce_mean(tf.reduce_sum(self.log_lik, axis=1))

                # with tf.variable_scope('local_gradients'):
                # entropy(for more exploration)
                self.entropy = -tf.reduce_mean(tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy,1e-10,1.)), axis=1))

                # with tf.variable_scope('critic_loss'):
                # Value loss
                # self.critic_loss = tf.reduce_mean(tf.square(self.td_error))
                self.critic_loss = tf.reduce_mean(tf.square(self.value - self.q_target), axis=1)

                # Total loss
                self.model_loss = self.actor_loss + self.critic_loss - self.entropy * 0.01

                # with tf.name_scope('local_gradients'):
                self.model_gradients = tf.gradients(self.model_loss, self.model_params) #calculate gradients for the network weights

            # with tf.name_scope('sync'): # update local and global network weights
            # with tf.name_scope('pull'):
            zipped_model_vars = zip(self.model_params, self.master_agent.model_params)
            self.pull_model_params_op = [l_a_p.assign(g_a_p) for l_a_p, g_a_p in zipped_model_vars]

            # with tf.name_scope('push'):
            zipped_model_vars = zip(self.model_gradients, self.master_agent.model_params)
            self.update_model_op = OPT_A.apply_gradients(zipped_model_vars)

    # neural network structure of the actor and critic
    def build_model(self, scope):

        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(0.1)

        with tf.variable_scope("model"):

            actor_hidden = tf.layers.dense(self.state, self.hidden1, tf.nn.tanh, kernel_initializer=w_init,
                                        bias_initializer=b_init)

            self.actor_predict = tf.layers.dense(actor_hidden, self.action_size, kernel_initializer=w_init,
                                                   bias_initializer=b_init)

            self.policy = tf.nn.softmax(self.actor_predict)
    
        # with tf.variable_scope("critic"):
            critic_hidden = tf.layers.dense(inputs=self.state, units = self.hidden1, activation=tf.nn.tanh,  # tanh activation
                kernel_initializer=w_init, bias_initializer=b_init, name='fc1_c')

            critic_predict = tf.layers.dense(inputs=critic_hidden, units = self.value_size, activation=None,
                kernel_initializer=w_init, bias_initializer=b_init, name='fc2_c')
            self.value = critic_predict
            
        self.model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/model')
        
        return self.policy, self.value, self.model_params # , self.critic_params

    def update_global(self, feed_dict):  # run by a local
        sess.run(self.update_model_op, feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        sess.run(self.pull_model_params_op)

    # get action from policy network
    def get_action(self, state):
        # Reshape observation to (num_features, 1)
        state_t = np.reshape(state, [1, self.state_size])
        # Run forward propagation to get softmax probabilities
        prob_weights = sess.run(self.policy, feed_dict={self.state: state_t})
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(np.arange(self.action_size), p=prob_weights[0])

        return action
        
# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, env, name, COORD, master_agent):
        self.env = env
        self.coordinator = COORD
        self.name = name
        self.agent = A3CAgent(name, master_agent)
        self.master_agent = master_agent

        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        # self.buffer_q_target = []
        self.action_size = action_size
        self.discount_factor = discount_factor
        
    # calculate discounted rewards
    def discount_and_norm_rewards(self):
        buffer_reward = np.vstack(self.buffer_reward)
        discounted_rewards = np.zeros_like(buffer_reward)
        running_add = 0
        for index in reversed(range(0, len(buffer_reward))):
            running_add = running_add * self.discount_factor + buffer_reward[index]
            discounted_rewards[index] = running_add
            
        # normalize episode rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def append_sample(self, state, action, reward):
        # Store actions as list of arrays
        # e.g. for action_size = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        action_array = np.zeros(self.action_size)
        action_array[action] = 1
        
        self.buffer_state.append(state)
        self.buffer_action.append(action_array)        
        self.buffer_reward.append(reward)

    # update policy network and value network every episode
    def train_model(self):
        discounted_rewards = self.discount_and_norm_rewards()
                    
        feed_dict={
            self.agent.state: np.vstack(self.buffer_state),
            self.agent.action: np.vstack(self.buffer_action),
            self.agent.q_target: discounted_rewards,
        }
        
        self.agent.update_global(feed_dict) # actual training step, update global A3CAgent
        self.agent.pull_global() # get global parameters to local A3CAgent    
        
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []

    def work(self):
        global episode, step
        
        avg_score = 10000
        episodes, scores = [], []

        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        
        # Step 3.2: run the game
        display_time = datetime.datetime.now()
        print("\n\n",game_name, "-game start at :",display_time,"\n")
        start_time = time.time()
        
        while time.time() - start_time < training_time and avg_score > 90:
            
            state = self.env.reset()

            done = False
            score = 10000
            ep_step = 0

            while not done and ep_step < ep_trial_step:
                # every time step we do train from the replay memory
                ep_step += 1
                step += 1
                
                # if self.name == 'W_0':
                #     self.env.render()
                
                # Select action_arr
                action = self.agent.get_action(state)
                
                # make step in environment
                next_state, reward, done, _ = self.env.step(action) 
                
                # save the sample <state, action, reward> to the memory
                self.append_sample(state, action, reward)

                score = ep_step

                state = next_state
                
                if done or ep_step == ep_trial_step:
                    episode += 1
                    self.train_model()
                    
                    # every episode, plot the play time
                    scores.append(score)
                    episodes.append(episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])

                    print('episode :{:>6,d}'.format(episode),'/ ep step :{:>5,d}'.format(ep_step), \
                          '/ time step :{:>8,d}'.format(step),'/ last 30 avg :{:> 4.1f}'.format(avg_score) )

                    break

        e = int(time.time() - start_time)
        print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

if __name__ == "__main__":
    sess = tf.Session()

    OPT_A = tf.train.AdamOptimizer(learning_rate, name='AdamOptimizer')
    global_agent = A3CAgent("master")  # we only need its params
    workers = []
    
    COORD = tf.train.Coordinator()
    
    # Create workers
    for index in range(N_WORKERS):
        env = gym.make(env_name).unwrapped
        i_name = 'W_%i' % index   # worker name
        workers.append(Worker(env, i_name, COORD, global_agent))

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
    

