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

env_name = 'MountainCar-v0'
# set environment
env = gym.make(env_name)
# env.seed(1)     # reproducible, general Policy gradient has high variance
# env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

training_time = 40*60
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
                self.actor_params, self.critic_params = self.build_model(scope)[-2:]
                
        else:   # local net, calculate losses
            with tf.variable_scope(self.scope):
                self.state = tf.placeholder(tf.float32,  [None, self.state_size], name='state')
                self.action = tf.placeholder(tf.int32,   [None, ],               name='action')
                self.q_target = tf.placeholder(tf.float32, [None, 1],          name='q_target')

                self.policy, self.value, self.actor_params, self.critic_params = self.build_model(scope)
                
                # with tf.variable_scope('td_error'):
                self.td_error = tf.subtract(self.q_target, self.value, name='td_error')

                # with tf.variable_scope('critic_loss'):
                self.critic_loss = tf.reduce_mean(tf.square(self.td_error))

                # with tf.variable_scope('actor_loss'):
                log_prob = tf.reduce_sum(tf.log(self.policy + 1e-5) * tf.one_hot(self.action, self.action_size, dtype=tf.float32), axis=1, keep_dims=True)
                exp_v = log_prob * tf.stop_gradient(self.td_error)
                entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-5),
                                         axis=1, keep_dims=True)  # encourage exploration
                self.exp_v = 0.001 * entropy + exp_v
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
            self.update_actor_op = OPT_A.apply_gradients(zipped_actor_vars)
            self.update_critic_op = OPT_C.apply_gradients(zipped_critic_vars)

    # neural network structure of the actor and critic
    def build_model(self, scope):

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
        
        return self.policy, self.value, self.actor_params, self.critic_params

    def update_global(self, feed_dict):  # run by a local
        sess.run([self.update_actor_op, self.update_critic_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        sess.run([self.pull_actor_params_op, self.pull_critic_params_op])

    # get action from policy network
    def get_action(self, state):
        # Reshape observation to (num_features, 1)
        state_t =  state[np.newaxis, :]
        # Run forward propagation to get softmax probabilities
        prob_weights = sess.run(self.policy, feed_dict={self.state: state_t})
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
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
            value_next_state = sess.run(self.agent.value, {self.agent.state: next_state[np.newaxis, :]})[0, 0]
            
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
        global episode, step
        
        avg_score = 10000
        episodes, scores = [], []

        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        
        # Step 3.2: run the game
        display_time = datetime.datetime.now()
        print("\n\n",game_name, "-game start at :",display_time,"\n")
        start_time = time.time()
        
        while time.time() - start_time < training_time and avg_score > 200:
            
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
                
                if step % 10 == 0 or done:   # update global and assign to local net
                    self.train_model(next_state, done)
                    
                score = ep_step

                state = next_state
                
                # train when epsisode finished
                if done or ep_step == ep_trial_step:
                    episode += 1
                    # self.train_model(next_state, done)
                    
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

    OPT_A = tf.train.RMSPropOptimizer(learning_rate, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(learning_rate, name='RMSPropC')
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
    
