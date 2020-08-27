#!/usr/bin/env python

import hmmlearn
import numpy as np
from scipy.misc import logsumexp
import rospy
from semantic_gaze_labeling.msg import LabeledFixationDataPoint

class GazeBasedPredictor:
    DT = 150 # CHECK THESE
    NMAX = 3


    def __init__(self, label_remap, goal_remaps, p_prior, p_trans, p_emiss):
        self.label_remap = label_remap
        self.goal_remaps = goal_remaps
        self.model = hmmlearn.hmm.MultinomialHMM( n_components=len(p_prior) )
        self.model.startprob_ = p_prior
        self.model.transmat_ = p_trans
        self.model.emissionprob_ = p_emiss

        self.reset()


    def update(self, label, duration):
        self._update_seq_with_label(label, duration)
        self._update_prob_for_seq()

    def reset(self):
        self._seq = []
        self._prob = np.full( (self.model.n_components,), 1./self.model.n_components ) # start with uniform prob


    def _update_seq_with_label(self, label, duration):
        ct = max(min( int(label/GazeBasedPredictor.DT), GazeBasedPredictor.NMAX), 1)
        self._seq += [self.label_remap[label]] * ct

    def _update_prob_for_seq(self):
        raw_log_prob = [ self.model.score( np.reshape(remap[self.seq], (-1, 1)) ) for remap in self.goal_remaps ]
        log_prob = raw_log_prob - logsumexp(raw_log_prob)
        self._prob = np.log(log_prob)

    # guard against setting from outside
    @property
    def prob(self):
        return self._prob

    @property
    def seq(self):
        return self._seq


class GazeBasedPredictorWrapper:
    def __init__(self, labeled_gaze_topic, *args, **kwargs):
        self.predictor = GazeBasedPredictor(*args, **kwargs)
        self.sub = rospy.Subscriber(labeled_gaze_topic, LabeledFixationDataPoint, self.labeled_fix_callback)

    def labeled_fix_callback(self, msg):
        # Do we need to guarantee that we receive these in sequence order or something?
        self.predictor.update(msg.label, msg.duration)

    def get_distribution(self):
        return self.predictor.prob


def load_gaze_predictor_from_params():
    labeled_gaze_topic = rospy.get_param('~labeled_gaze_topic', '/semantic_gaze_labeler/output')

    # none -> none, [mico_link_x] -> robot, [ee|fork] -> ee, morsels separate
    default_remap = np.array([0,1,1,1,1,1,1,1,2,2,3,4,5]
    default_goal_remaps = [ np.range(6) ] * 3
    for i, r in enumerate(default_goal_remaps):
        r[r >= 3 & r != 3+i] = 4 # remap incorrect labels -> 4
        r[3+i] = 3 # remap correct -> 3

    label_remap = rospy.get_param('~label_remap', default_remap)
    goal_remaps = rospy.get_param('~goal_remaps', default_goal_remaps)

    p_prior = np.array( [] ).reshape((-1, 1))
    p_trans = np.array( [] )
    p_emiss = np.array( [] ).reshape((-1, 1))

    return GazeBasedPredictorWrapper(labeled_gaze_topic, 
        label_remap=label_remap, goal_remaps=goal_remaps,
        p_prior=p_prior, p_trans=p_trans, p_emiss=p_emiss)
