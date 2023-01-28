import tensorflow_probability as tfp
import tensorflow as tf

#cold days as 0, hot days as 1
#first day sequence has a chance of 80% to be cold
#cold day has 30% chance of being followed by a hot day
#hot day has 20% chance of being followed by a cold day
#mean - vidurkis and standard deviaton - kiek elementai nutole nuo vidurkio vidutiniskai
#cold day mean 0 standard deviation 5, hot day mean 15 and standard deviation 10

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) #pirma diena 0.8 arba 0.2
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]]) #two states, two probabilities, pirma cold day, antra hot day

observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.]) #loc -- mean, scale -- standard deviation

model = tfp.distributions.HiddenMarkovModel(initial_distribution=initial_distribution, transition_distribution= transition_distribution, observation_distribution= observation_distribution, num_steps= 50)

mean = model.mean() #partialy defined
# with tf.compat.v1.Session() as sess:
#     print(mean.numpy())

print(mean)