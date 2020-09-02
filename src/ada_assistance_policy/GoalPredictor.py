import numpy as np
import scipy.misc
import IPython

logsumexp = scipy.misc.logsumexp

class PolicyBasedGoalPredictor:
  def __init__(self, rl_policy):
    self.rl_policy = rl_policy
    self.discount_factor = 1
    self.log_goal_distribution = np.log(
        (1./len(self.rl_policy.goals))*np.ones(len(self.rl_policy.goals)))

  def update(self):
    # Normally we would need to update rl_policy with the new values
    # BUT we're using the same policy object to calculate action proposals
    # So it's updated in the main thread
    # So we can just access the values
    values, q_values = self.rl_policy.get_novalues()
    log_goal_distribution_unnormed =  self.discount_factor * self.log_goal_distribution - (q_values - values)
    self.log_goal_distribution = normalize_and_clip(
        log_goal_distribution_unnormed)

  def get_log_distribution(self):
    return self.log_goal_distribution

  def get_distribution(self):
    return np.exp(self.log_goal_distribution)



class MergedGoalPredictor:
    def __init__(self, predictors, weights=None):
        self.predictors = predictors
        self.weights = weights if weights is not None else np.ones( (len(predictors)) )
    
    def update():
        # just pass on to underlying functions
        for pred in self.predictors:
            pred.update()

    def get_log_distribution():
        sub_log_distribs = np.array(
            [pred.get_log_distribution() for pred in self.predictors])
        log_distribs = np.dot(self.weights, sub_log_distribs)
        return normalize_and_clip(log_distribs)

    def get_distribution(self):
        return np.exp(self.get_log_distribution())
        


## Helper functions
def normalize_and_clip(log_distribution_unnormed):
    return clip_probability(log_distribution_unnormed - logsumexp(log_distribution_unnormed))

MAX_PROB_ANY_GOAL = 0.99
LOG_MAX_PROB_ANY_GOAL = np.log(MAX_PROB_ANY_GOAL)
def clip_probability(log_goal_distribution):
    if len(log_goal_distribution) <= 1:
        return log_goal_distribution
    #check if any too high
    max_prob_ind = np.argmax(log_goal_distribution)
    if log_goal_distribution[max_prob_ind] > LOG_MAX_PROB_ANY_GOAL:
        #see how much we will remove from probability
        diff = np.exp(
            log_goal_distribution[max_prob_ind]) - MAX_PROB_ANY_GOAL
        #want to distribute this evenly among other goals
        diff_per = diff/(len(log_goal_distribution)-1.)

        #distribute this evenly in the probability space...this corresponds to doing so in log space
        # e^x_new = e^x_old + diff_per, and this is formulate for log addition
        log_goal_distribution += np.log(1. +
                diff_per/np.exp(log_goal_distribution))
        #set old one
        log_goal_distribution[max_prob_ind] = LOG_MAX_PROB_ANY_GOAL
    return log_goal_distribution
