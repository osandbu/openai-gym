import gym
import np
import numpy

# hyperparameters
episode_number = 0
batch_size = 10
gamma = 0.99 # discount factor for reward
decay_rate = 0.99
num_hidden_layer_neurons = 200
n = 80
input_dimensions = 80 * 80
learning_rate = 1e-4

def average_color(a,b,c,d,i):
    return (int(a[0]) + int(b[0]) + int(c[0]) + int(d[0])) / 4

# [y for y in [x for x in a]
# from 160x160 to 80x80
# observation :: [160,160,3]
# output :: [80,80,3]
def downsample(observation):
    output = np.zeros((n,n,3))
    for x in range(0,n):
      for y in range(0,n):
        a = observation[2*x,2*y]
        b = observation[2*x,2*y+1]
        c = observation[2*x+1,2*y]
        d = observation[2*x+1,2*y+1]
        output[x,y]= [
            average_color(a,b,c,d,0),
            average_color(a,b,c,d,1),
            average_color(a,b,c,d,2)
        ]
    return output

# from 0.33 to 0, from 0.66 to 1
# observation :: [80,80,3]
# output :: [80,80]
def remove_color(observation):
    output = np.zeros((n,n))
    for x in range(0,n):
      for y in range(0,n):
          color = observation[x,y]
          grayscale = (color[0] + color[1] + color[2]) / 3
          output[x,y] = grayscale
    return output

# observation :: [80,80]
# output :: [80,80]
def remove_background(observation):
    return observation

# input: [6400,1]
# output: [1,6400]
def relu(input):
    return numpy.transpose(input)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def choose_action(up_probability):
    return int(up_probability + 0.5)

def discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma):
    return episode_gradient_log_ps
    #return episode_gradient_log_ps_discounted

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)

    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    distinct = []
    for x in processed_observation:
        if x not in distinct:
            distinct.append(x)

    for x in distinct:
        print x

    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations

def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(weights['2'], hidden_layer_values)
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    """ See here: http://neuralnetworksanddeeplearning.com/chap2.html"""
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    """ See here: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

def main():
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)

        episode_observations = numpy.append(episode_observations, processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        # Combine the following values for the episode
        episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
        episode_observations = np.vstack(episode_observations)
        episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
        episode_rewards = np.vstack(episode_rewards)

        # Tweak the gradient of the log_ps based on the discounted rewards
        episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)
        gradient = compute_gradient(
            episode_gradient_log_ps_discounted,
            episode_hidden_layer_values,
            episode_observations,
            weights
        )

        if episode_number % batch_size == 0:
            update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

main()
