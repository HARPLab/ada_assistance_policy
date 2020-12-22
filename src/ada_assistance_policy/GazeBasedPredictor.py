#!/usr/bin/env python

import hmmlearn.hmm
import numpy as np
try:
    import rospy
    from semantic_gaze_labeling.msg import LabeledFixationDataPoint
except ImportError:
    rospy = None
    LabeledFixationDataPoint = None
import threading

from ada_assistance_policy.GoalPredictor import normalize_and_clip

class SequenceProcessor:
    def __init__(self, label_remap, goal_remaps, dt, nmax):
        self.label_remap = label_remap
        self.goal_remaps = goal_remaps
        self.dt = dt
        self.nmax = nmax

    def process_item(self, duration, label, goal):
        ct = max(min( int(duration/GazeBasedPredictor.DT), GazeBasedPredictor.NMAX), 1)
        lbl = self.goal_remaps[goal][self.label_remap[label]]
        return [lbl] * ct

    def process(self, seq, goal):
        return sum((self.process_item(*item, goal=goal) for item in seq), [])

    def get_prob(self, seq, model):
        processed_seq = [ self.process(seq, goal) for goal in range(len(self.goal_remaps)) ]
        scores = np.array([ model.score(np.array(seq).reshape(-1, 1)) for seq in processed_seq ])
        return normalize_and_clip(scores)

    def fit(self, seqs, model):
        processed_seqs = [ self.process(seq, goal) for seq, goal in seqs ]
        x, lengths = np.reshape(np.concatenate(processed_seqs), (-1, 1)), [ len(s) for s in processed_seqs ]
        model.fit(x, lengths)
        return model

        

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

    label_remap = np.array(rospy.get_param('~params/label_remap'))
    goal_remaps = np.array(rospy.get_param('~params/goal_remaps'))

    n_components = rospy.get_param('~params/n_components')
    dt = rospy.get_param('~params/dt')
    n_max = rospy.get_param('~params/n_max')

    startprob = np.array(rospy.get_param('~model/startprob'))
    transmat = np.array(rospy.get_param('~model/transmat'))
    emissionprob = np.array(rospy.get_param('~model/emissionprob'))

    processor = SequenceProcessor(label_remap, goal_remaps, dt, n_max)
    model = hmmlearn.hmm.MultinomialHMM(n_components=n_components)
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob
    # make sure the parameters match
    model._check()

    return GazeBasedPredictorWrapper(labeled_gaze_topic, 
        label_remap=label_remap, goal_remaps=goal_remaps,
        p_prior=p_prior, p_trans=p_trans, p_emiss=p_emiss)
