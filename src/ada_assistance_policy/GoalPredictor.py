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
    self.type = 'policy'

  def update(self):
    # Normally we would need to update rl_policy with the new values
    # BUT we're using the same policy object to calculate action proposals
    # So it's updated in the main thread
    # So we can just access the values
    values, q_values = self.rl_policy.get_novalues()
    log_goal_distribution_unnormed =  self.discount_factor * self.log_goal_distribution - (q_values - values)
    self.log_goal_distribution = normalize_and_clip(
        log_goal_distribution_unnormed)

  def get_log_distribution(self, get_log=False):
    if get_log:
        return self.log_goal_distribution, {}
    else:
        return self.log_goal_distribution

  def get_distribution(self, get_log=False):
    if get_log:
        return np.exp(self.log_goal_distribution), {}
    else:
        return np.exp(self.log_goal_distribution)

  def get_config(self):
      return { 'type': self.type }


def _array_to_str(arr):
    return np.array2string(arr, precision=6).replace('\n', '')


class MergedGoalPredictor:
    def __init__(self, predictors, weights=None):
        self.predictors = predictors
        self.weights = weights if weights is not None else np.ones( (len(predictors)) )
        self.type = 'merged[{}]'.format(', '.join(p.type for p in predictors))
    
    def update(self):
        # just pass on to underlying functions
        for pred in self.predictors:
            pred.update()

    def get_log_distribution(self, get_log=False):
        sub_data = [pred.get_log_distribution(get_log) for pred in self.predictors]
        if get_log:
            sub_log_distribs = np.array([d[0] for d in sub_data])
        else:
            sub_log_distribs = np.array(sub_data)
        log_distribs = np.dot(self.weights, sub_log_distribs)
        log_dist = normalize_and_clip(log_distribs)
        if get_log:
            # get a flattened list of underlying probabilities
            sub_dists = {}
            for i, p in enumerate(sub_log_distribs):
                sub_dists['sub{}_p'.format(i)] = _array_to_str(p)
                for k, val in sub_data[i][1].items():  # get the log info from the underlying dist
                    sub_dists['sub{}_{}'.format(i, k)] = val
            return log_dist, sub_dists
        else:
            return log_dists


    def get_distribution(self, get_log=False):
        if get_log:
            log_dist, log = self.get_log_distribution(True)
            return np.exp(log_dist), log
        else:
            return np.exp(self.get_log_distribution(False))

    def get_config(self):
        cfg = { 'type': self.type }
        for i, p in enumerate(self.predictors):
            p_cfg = p.get_config()
            p_cfg['weight'] = float(self.weights[i])
            cfg['p{}'.format(i)] = p_cfg
        return cfg
        


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
