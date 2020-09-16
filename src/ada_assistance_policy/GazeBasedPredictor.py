#!/usr/bin/env python

import hmmlearn.hmm
import numpy as np
from scipy.misc import logsumexp
import rospy
import threading

from semantic_gaze_labeling.msg import LabeledFixationDataPoint
from GoalPredictor import normalize_and_clip

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
        self._lock = threading.RLock()

        self.reset()

    def process_message(self, label, duration):
        with self._lock:
            self._update_seq_with_label(label, duration)
            self._update_prob_for_seq()

    def reset(self):
        self._seq = []
        prob = np.full( (self.model.n_components,), 1./self.model.n_components ) # start with uniform prob
        self._log_prob = np.log(prob)

    def _update_seq_with_label(self, label, duration):
        ct = max(min( int(label/GazeBasedPredictor.DT), GazeBasedPredictor.NMAX), 1)
        self._seq += [self.label_remap[label]] * ct

    def _update_prob_for_seq(self):
        # TODO: could do a much fancier incremental thing from the model which would run faster
        # like, the point of HMMs is that you can do this efficiently without starting from nothing
        # but since (1) labels only come in at like ~1-2 Hz and (2) sequences are pretty short (<100)
        # almost certainly not worth it
        raw_log_prob = [ self.model.score( np.reshape(remap[self._seq], (-1, 1)) ) for remap in self.goal_remaps ]
        self._log_prob = normalize_and_clip(raw_log_prob)

    def get_log_distribution(self, get_log=False):
        with self._lock:  # technically it's only assignment that crosses thread bounds so it should be ok
                          # but use a lock so we can e.g. log info about the sequence at the time of prob retrieval
            if get_log:
                return self._log_prob, {'n_seq': len(self._seq)}
            else:
                return self._log_prob

    def get_config(self):
        return { 
            'startprob': self.model.startprob_.tolist(),
            'transmat': self.model.transmat_.tolist(),
            'emissionprob': self.model.emissionprob_.tolist()
        }


class GazeBasedPredictorWrapper:
    def __init__(self, labeled_gaze_topic, *args, **kwargs):
        self.predictor = GazeBasedPredictor(*args, **kwargs)
        self.sub = rospy.Subscriber(labeled_gaze_topic, LabeledFixationDataPoint, self.labeled_fix_callback)
        self.type = 'gaze'

    def labeled_fix_callback(self, msg):
        # Do we need to guarantee that we receive these in sequence order or something?
        self.predictor.process_message(msg.label, msg.duration)

    def update(self):
        # called from main (joystick) thread
        # but we're running separately
        # so don't do anything
        pass

    def get_log_distribution(self, get_log=False):
        return self.predictor.get_log_distribution(get_log)

    def get_distribution(self, get_log=False):
        if get_log:
            log_dist, log = self.get_log_distribution(True)
            return np.exp(log_dist), log
        else:
            return np.exp(self.get_log_distribution(False))

    def get_config(self):
        cfg = { 
            'type': self.type,
        }
        cfg.update(self.predictor.get_config())
        return cfg


def load_gaze_predictor_from_params():
    labeled_gaze_topic = rospy.get_param('~labeled_gaze_topic', '/semantic_gaze_labeler/output')

    # none -> none, [mico_link_x] -> robot, [ee|fork] -> ee, morsels separate
    default_remap = np.array([0,1,1,1,1,1,1,1,2,2,3,4,5])
    default_goal_remaps = [ np.arange(6) ] * 3
    for i, r in enumerate(default_goal_remaps):
        r[np.logical_and(r >= 3, r != 3+i)] = 4 # remap incorrect labels -> 4
        r[3+i] = 3 # remap correct -> 3

    # get_param doesn't like np.array for lists
    # but we like the logical indexing above
    # so unwrap and rewrap in np.array
    label_remap = np.array(rospy.get_param('~label_remap', default_remap.tolist()))
    goal_remaps = np.array(rospy.get_param('~goal_remaps', [r.tolist() for r in default_goal_remaps] ))

    p_prior = np.array( [1, 0, 0] ).reshape((-1, 1))
    p_trans = np.array( [[1, 0, 0], [0, 1, 0], [0, 0, 1]] )
    p_emiss = np.array( [[0.4, 0.3, 0.3, 0, 0], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]] ).reshape((-1, 1))

    return GazeBasedPredictorWrapper(labeled_gaze_topic, 
        label_remap=label_remap, goal_remaps=goal_remaps,
        p_prior=p_prior, p_trans=p_trans, p_emiss=p_emiss)
