import os
import sys
import gym
import pylab
import numpy as np
import time
import tensorflow as tf

env_name = "CartPole-v1"
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance
# env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

MAX_EP_STEP = 500
ENTROPY_BETA = 0.001

model_lr = 0.005

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

# Network for the Actor Critic
class A2C_agent(object):
    def __init__(self, sess, scope):
        self.sess = sess
        # get size of state and action
        self.action_size = action_size
        self.state_size = state_size
        self.value_size = 1
        
        # these is hyper parameters for the ActorCritic
        self.discount_factor = 0.99         # decay rate
        
        self.hidden1, self.hidden2 = 128, 128
        
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
        self.q_target = tf.placeholder(tf.float32, name="q_target")

    def _init_op(self):
        # with tf.variable_scope('td_error'):
        # A_t = R_t - V(S_t)
        # self.td_error = tf.subtract(self.q_target, self.value, name='td_error')
        self.td_error = self.q_target - self.value
        
        # with tf.variable_scope('critic_loss'):
        # Value loss
        # self.critic_loss = tf.reduce_mean(tf.square(self.td_error))
        self.critic_loss = tf.reduce_mean(tf.square(self.value - self.q_target), axis=1)

        # with tf.variable_scope('actor_loss'):
        log_prob = tf.reduce_sum(tf.log(self.policy + 1e-5) * tf.one_hot(self.action, self.action_size, dtype=tf.float32), axis=1, keep_dims=True)
        exp_v = log_prob * tf.stop_gradient(self.td_error)
        entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-5),
                                 axis=1, keep_dims=True)  # encourage exploration
        self.exp_v = ENTROPY_BETA * entropy + exp_v
        self.actor_loss = tf.reduce_mean(-self.exp_v)
        
        self.loss_total = self.actor_loss + self.critic_loss
        
        self.model_gradients = tf.gradients(self.loss_total, self.model_params) #calculate gradients for the network weights
        self.model_optimizer = tf.train.RMSPropOptimizer(model_lr)
        
        zipped_model_vars = zip(self.model_gradients, self.model_params)
        self.update_model_op = self.model_optimizer.apply_gradients(zipped_model_vars)
        
    # neural network structure of the actor and critic
    def build_model(self):

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
            value_next_state = self.sess.run(self.value, {self.state: next_state[np.newaxis, :]})[0][0]

        for reward in self.buffer_reward[::-1]:    # reverse buffer r
            value_next_state = reward + self.discount_factor * value_next_state
            self.buffer_q_target.append(value_next_state)

        self.buffer_q_target.reverse()

        feed_dict={
            self.state: np.vstack(self.buffer_state),
            self.action: np.array(self.buffer_action),
            self.q_target: np.vstack(self.buffer_q_target)
        } 
        
        self.sess.run(self.update_model_op, feed_dict)
        
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        self.buffer_q_target = []


def main():
    with tf.Session() as sess:
        agent = A2C_agent(sess, "model")

        agent.sess.run(tf.global_variables_initializer())
        train_steps = 0
        
        agent.buffer_state, agent.buffer_action, agent.buffer_reward = [], [], []
        agent.buffer_q_target = []
        
        scores, episodes = [], []
        episode = 0
        avg_score = 0

        start_time = time.time()
        
        while time.time() - start_time < 5*60 and avg_score < 495:
            
            done = False
            score = 0
            state = env.reset()

            while not done and score < MAX_EP_STEP:
                # every time step we do train from the replay memory
                score += 1
                
                # fresh env
                # if agent.render:
                #     env.render()
                train_steps += 1
                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                
                # make step in environment
                next_state, reward, done, _ = env.step(action) 
                
                # save the sample <state, action, reward> to the memory
                agent.append_sample(state, action, reward)
                
                if train_steps % 10 == 0 or done:   # update global and assign to local net
                    agent.train_model(next_state, done)
                    
                # swap observation
                state = next_state
                
                # train when epsisode finished
                if done or score == MAX_EP_STEP:
                    episode += 1
                    # agent.train_model(next_state, done)
                    
                    # every episode, plot the play time
                    scores.append(score)
                    episodes.append(episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])
                    
                    print("episode :{:5d}".format(episode), "/ score :{:5d}".format(score))
                    
                    break

        pylab.plot(episodes, scores, 'b')
        pylab.savefig("./save_graph/Cartpole_PG_TF.png")
        e = int(time.time() - start_time)
        print('Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        sys.exit()

if __name__ == "__main__":
    main()
