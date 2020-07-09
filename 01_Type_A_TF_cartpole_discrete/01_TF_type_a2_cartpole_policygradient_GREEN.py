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
model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)
    
# This is REINFORCE agent for the Cartpole
class PolicyGradient:
    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        
        # get size of state and action
        self.action_size = action_size
        self.state_size = state_size
        # these is hyper parameters for the PolicyGradient
        self.discount_factor = 0.99         # decay rate
        self.model_lr = 0.005
        self.hidden1, self.hidden2 = 128, 128
        
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        
        self._init_input()
        self.build_model()
        self._init_op()

    def _init_input(self):
        # with tf.variable_scope('input'):
        self.state = tf.placeholder(tf.float32,  [None, self.state_size], name='state')
        self.action = tf.placeholder(tf.float32, [None, self.action_size], name="action")
        self.reward = tf.placeholder(tf.float32, name="reward")

    def _init_op(self):
        # with tf.variable_scope('actor_loss'):
        self.log_p = self.action * tf.log(self.policy)
        self.log_lik = self.log_p * self.reward
        self.actor_loss = tf.reduce_mean(tf.reduce_sum(-self.log_lik, axis=1))
        
        # with tf.variable_scope('train'):
        self.train_op = tf.train.AdamOptimizer(self.model_lr).minimize(self.actor_loss)
                    
    # neural network structure of the actor and critic
    def build_model(self):

        w_init, b_init = tf.random_normal_initializer(mean=0, stddev=0.3), tf.constant_initializer(0.1)
        # fc1
        actor_hidden = tf.layers.dense(inputs=self.state, units = self.hidden1, activation=tf.nn.tanh,  # tanh activation
            kernel_initializer = w_init, bias_initializer = b_init, name='fc1_a')
        # fc2
        actor_predict = tf.layers.dense(inputs=actor_hidden, units = self.action_size, activation=None,
            kernel_initializer = w_init, bias_initializer = b_init, name='fc2_a')

        self.policy = tf.nn.softmax(actor_predict, name='act_prob')  # use softmax to convert to probability
        
    # get action from policy network
    def get_action(self, state):
        """
            Choose action based on observation

            Arguments:
                state: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        state_t = np.reshape(state, [1, self.state_size])
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.policy, feed_dict={self.state: state_t})
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(np.arange(self.action_size), p=prob_weights[0])

        return action
        
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
    
    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def append_sample(self, state, action, reward):
        # Store actions as list of arrays
        # e.g. for action_size = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        act = np.zeros(self.action_size)
        act[action] = 1
        
        self.buffer_state.append(state)
        self.buffer_action.append(act)        
        self.buffer_reward.append(reward)
   
    # update policy network and value network every episode
    def train_model(self):
        discounted_rewards = self.discount_and_norm_rewards()
                    
        feed_dict={
            self.state: np.vstack(self.buffer_state),
            self.action: np.vstack(self.buffer_action),
            self.reward: discounted_rewards,
        }
        
        actor_loss,_ = self.sess.run([self.actor_loss, self.train_op], feed_dict)
        
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []

def main():
    with tf.Session() as sess:
        agent = PolicyGradient(sess, state_size, action_size)

        agent.sess.run(tf.global_variables_initializer())
        train_steps = 0
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
                
                # if train_steps % 10 == 0 or done:   # update global and assign to local net
                #     agent.train_model()
                    
                # swap observation
                state = next_state
                
                # train when epsisode finished
                if done or score == MAX_EP_STEP:
                    episode += 1
                    agent.train_model()
                
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