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

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
import tkFileDialog
import yaml

from ada_assistance_policy.GoalPredictor import normalize_and_clip

class SequenceProcessor:
    def __init__(self, label_remap, goal_remaps, dt, nmax):
        self.label_remap = label_remap
        self.goal_remaps = goal_remaps
        self.dt = dt
        self.nmax = nmax

    def process_item(self, duration, label, goal):
        ct = max(min( int(duration/self.dt), self.nmax), 1)
        lbl = self.goal_remaps[goal][self.label_remap[label+1]]  # offset -1 label to 0
        return [lbl] * ct

    def process(self, seq, goal):
        return sum((self.process_item(*item, goal=goal) for item in seq), [])

    def get_prob(self, seq, model):
        processed_seqs = [ self.process(seq, goal) for goal in range(len(self.goal_remaps)) ]
        return self.get_prob_from_processed(processed_seq, model)

    def get_prob_from_processed(self, processed_seqs, model):
        scores = np.array([ model.score(np.array(seq).reshape(-1, 1)) for seq in processed_seqs ])
        return normalize_and_clip(scores)

    def fit(self, seqs, model):
        processed_seqs = [ self.process(seq, goal) for seq, goal in seqs ]
        x, lengths = np.reshape(np.concatenate(processed_seqs), (-1, 1)), [ len(s) for s in processed_seqs ]
        model.fit(x, lengths)
        return model

    @property
    def num_goals(self):
        return len(self.goal_remaps)

        

class GazeBasedPredictor:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self._lock = threading.RLock()

        self.reset()

    def process_message(self, label, duration):
        with self._lock:
            self._update_seq_with_label(label, duration)
            self._update_prob_for_seq()

    def reset(self):
        self._seqs = [ list() for _ in range(self.processor.num_goals) ]
        prob = np.full( (self.processor.num_goals,), 1./self.processor.num_goals ) # start with uniform prob
        self._log_prob = np.log(prob)

    def _update_seq_with_label(self, label, duration):
        rospy.logdebug("cur seqs: {}".format(self._seqs))
        for seq, goal in zip(self._seqs, range(self.processor.num_goals)):
            items = self.processor.process_item(duration, label, goal)
            rospy.logdebug("extend ({}, {}) -> [{}] {}".format(label, duration, goal, items))
            seq.extend(items)

    def _update_prob_for_seq(self):
        # TODO: could do a much fancier incremental thing from the model which would run faster
        # like, the point of HMMs is that you can do this efficiently without starting from nothing
        # but since (1) labels only come in at like ~1-2 Hz and (2) sequences are pretty short (<100)
        # almost certainly not worth it
        self._log_prob = self.processor.get_prob_from_processed(self._seqs, self.model)
        rospy.logdebug("updated prob to {}".format(self._log_prob))

    def get_log_distribution(self, get_log=False):
        with self._lock:  # technically it's only assignment that crosses thread bounds so it should be ok
                          # but use a lock so we can e.g. log info about the sequence at the time of prob retrieval
            if get_log:
                return self._log_prob, {'n_seq': len(self._seqs[0])}  # TODO: more log stuff?
            else:
                return self._log_prob

    def get_config(self):
        return { 
            'startprob': self.model.startprob_.tolist(),
            'transmat': self.model.transmat_.tolist(),
            'emissionprob': self.model.emissionprob_.tolist()
        }


class GazeBasedPredictorWrapper:
    def __init__(self, predictor, labeled_gaze_topic):
        rospy.loginfo("Connected gaze predictor to {}".format(labeled_gaze_topic))
        self.predictor = predictor
        self.sub = rospy.Subscriber(labeled_gaze_topic, LabeledFixationDataPoint, self.labeled_fix_callback)
        self.type = 'gaze'

    def labeled_fix_callback(self, msg):
        rospy.loginfo("got message: {}, {}".format(msg.label, msg.duration))
        # Do we need to guarantee that we receive these in sequence order or something?
        # use messagefilters for this (though not in python yet)
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


GAZE_BASED_PREDICTOR_CONFIG_NAME = "gaze_based_predictor"

class GazeBasedPredictorConfigFrame(tk.LabelFrame, object):
    def __init__(self, parent, initial_config={}):
        super(GazeBasedPredictorConfigFrame, self).__init__(parent, text="Gaze-based prediction options")
        initial_config = initial_config.get(GAZE_BASED_PREDICTOR_CONFIG_NAME, {})

        self._enabled_var = tk.IntVar()
        self._enabled_var.set(initial_config.get("enabled", False))
        self._enabled_check = tk.Checkbutton(self, text="Enabled", variable=self._enabled_var)
        self._enabled_check.grid(row=0, column=0, stick="nw")

        self._topic_label = tk.Label(self, text="Topic to receive labeled gaze:")
        self._topic_var = tk.StringVar()
        self._topic_var.set(initial_config.get("labeled_gaze_topic", "/semantic_gaze_labeler/output"))
        self._topic_entry = tk.Entry(self, textvariable=self._topic_var)
        self._topic_label.grid(row=1, column=0, sticky="nw")
        self._topic_entry.grid(row=2, column=0, sticky="new")

        self._model_title_label = tk.Label(self, text="Model to load:")
        self._model_loc_var = tk.StringVar()
        self._model_loc_var.set(initial_config.get("model_file", ""))
        self._model_loc_label = tk.Label(self, textvariable=self._model_loc_var, justify=tk.RIGHT)
        self._model_loc_label.bind("<Configure>", lambda e: e.widget.configure(wraplength=e.widget.winfo_width()))
        self._model_loc_btn = tk.Button(self, text="Select", command=self._select_model_file)
        self._model_title_label.grid(row=3, column=0, sticky="nw")
        self._model_loc_label.grid(row=4, column=0, sticky="ne")
        self._model_loc_btn.grid(row=5, column=0, sticky="nw")

        self.rowconfigure(4, weight=1)
        self.columnconfigure(0, weight=1)

    def _select_model_file(self):
        fn = tkFileDialog.askopenfilename(title="Load which model file?")
        if fn is not None:
            try:
                _load_gaze_predictor_model(fn)
            except Exception as e:
                tk.messagebox.showerror("Error loading model", "Could not load model {}: {}".format(fn, str(e)))
            else:
                self._model_loc_var.set(fn)

    def get_config(self):
        return { GAZE_BASED_PREDICTOR_CONFIG_NAME: {
            "enabled": bool(self._enabled_var.get()),
            "labeled_gaze_topic": self._topic_var.get(),
            "model_file": self._model_loc_var.get()
        } }

    def set_state(self, state):
        self._enabled_check.configure(state=state)
        self._topic_entry.configure(state=state)
        self._model_loc_btn.configure(state=state)

def load_gaze_predictor(config):
    config = config.get(GAZE_BASED_PREDICTOR_CONFIG_NAME, {})
    if config.get("enabled", False):
        predictor = _load_gaze_predictor_model(config.get("model_file", ""))
        return GazeBasedPredictorWrapper(predictor, config.get("labeled_gaze_topic", ""))
    else:
        return None


def _load_gaze_predictor_model(fn):
    with open(fn, 'r') as f:
        data = yaml.load(f)

    label_remap = np.array(data['params']['label_remap'])
    goal_remaps = np.array(data['params']['goal_remaps'])

    n_components = data['params']['n_components']
    dt = data['params']['dt']
    n_max = data['params']['n_max']

    startprob = np.array(data['model']['startprob'])
    transmat = np.array(data['model']['transmat'])
    emissionprob = np.array(data['model']['emissionprob'])

    processor = SequenceProcessor(label_remap, goal_remaps, dt, n_max)
    model = hmmlearn.hmm.MultinomialHMM(n_components=n_components)
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob
    # make sure the parameters match
    model._check()

    return GazeBasedPredictor(model, processor)

