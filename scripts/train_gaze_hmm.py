#!/usr/bin/env python

import argparse
import functools
import hmmlearn
import itertools
import numpy as np
import os
import pandas as pd
import progressbar
import sklearn.base
import sklearn.model_selection
import yaml

from ada_assistance_policy.GazeBasedPredictor import SequenceProcessor


def load_data(data_dir, name):
    res = {'name': name, 'data_dir': data_dir}
    
    with open(os.path.join(data_dir, 'text_data', 'morsel.yaml'), 'r') as f:
        morsel_data = yaml.safe_load(f)
        res.update({'morsel_target': morsel_data['stated_goal'],
        'morsel_mask': [morsel_data['morsel{}'.format(i)] is not None for i in (0,1,2)],
        'goal_success': morsel_data['goal_success'],
        'any_success': morsel_data['any_success']})
    
    with open(os.path.join(data_dir, 'stats', 'run_info.yaml'), 'r') as f:
        run_info = yaml.safe_load(f)
        run_dict = {k:v for k,v in run_info}
        res.update({'run_info': run_dict})
        
    with open(os.path.join(data_dir, 'text_data', 'control_mode.txt'), 'r') as f:
        assistance_mode = f.read()[0]
        res.update({'assistance_mode': assistance_mode})
    
    fix_data = pd.read_csv(os.path.join(data_dir, 'processed', 'fixations.csv'))
    if 'label' in fix_data:
        fix_data = fix_data.assign(label = fix_data.label+1) # need to offset because multiclass hmm doesn't like -1 labels
    else:
        fix_data = None
    res.update({'fix_data': fix_data})
        
    try:
        assistance_data = pd.read_csv(os.path.join(data_dir, 'text_data', 'assistance_info.csv'))
        res.update({'assistance_data': assistance_data})
    except:
        res.update({'assistance_data': None})
    
    try:
        scene_kp = pd.read_csv(os.path.join(data_dir, 'processed', 'scene_keypoints.csv'))
        res.update({'scene_keypoints': scene_kp})
    except:
        res.update({'scene_keypoints': None})
    
    return res


def is_mode(x, modes):
    try:
        if not os.path.exists(os.path.join(x, 'processed', 'fixations.csv')):
            return False
        with open(os.path.join(x, 'text_data', 'control_mode.txt'), 'r') as f:
            return f.read()[0] in modes
    except IOError:
        return False

is_teleop = functools.partial(is_mode, modes='0')
    
def load_all_data(root_dir, filt=is_teleop):
    dirs = [(os.path.join(root_dir, p, 'run', x), '{}_{}'.format(p, x))
               for p in os.listdir(root_dir) if p[0] == 'p'
               for x in os.listdir(os.path.join(root_dir, p, 'run'))
               if filt(os.path.join(root_dir, p, 'run', x))]
    return [load_data(*d) for d in progressbar.progressbar(dirs)]

def data_to_seq(fix_data):
    return [ (f.duration, f.label) for f in fix_data.itertuples()]
def data_to_seqs(data):
    return [ (data_to_seq(d['fix_data']), d['morsel_target']) for d in data]

def is_correct(probs, idx):
    js = set(range(len(probs)))
    for j in js-{idx}:
        if probs[j] >= probs[idx]:
            return False
    return True

def score_result(seqs, log_probs):
    correct = np.array([ is_correct(prob, seq[1]) for prob, seq in zip(log_probs, seqs)])
    true_log_probs = np.array([ prob[seq[1]] for prob, seq in zip(log_probs, seqs) ])

    return {
        'accuracy': float(np.count_nonzero(correct)) / len(correct),
        'mean_prob': np.mean(np.exp(true_log_probs)),
        'median_prob': np.median(np.exp(true_log_probs))
    }


def eval_crossval(model, processor, seqs, n_splits=5):
    kf = sklearn.model_selection.KFold(n_splits=n_splits)
    scores = []
    for train, test in kf.split(seqs):
        m = sklearn.base.clone(model)
        m = processor.fit([ seqs[i] for i in train], m)
        seq_test = [ seqs[j] for j in test]
        log_probs = np.vstack([processor.get_prob(seq[0], m) for seq in seq_test])
        scores.append(score_result(seq_test, log_probs))
    return pd.DataFrame(scores).mean()



def eval_model(model, processor, seqs):
    model = processor.fit(seqs, model)
    log_probs = np.vstack([processor.get_prob(seq[0], model) for seq in seqs])
    return score_result(seqs, log_probs)


def main():
    parser = argparse.ArgumentParser(description='Train parameters for HMMs for gaze-based goal recognition')
    parser.add_argument('root_dir', help='Root directory for dataset to load')
    parser.add_argument('--n-components', type=int, nargs='+', default=[3], help='Components to use for HMM')
    parser.add_argument('--nmax', type=int, nargs='+', default=[3], help='Maximum repetitions from dt')
    parser.add_argument('--dt', type=float, nargs='+', default=[150.], help='dt for quantizing fixations')
    parser.add_argument('--method', choices=['crossval', 'best'], default='best', help='evaluation method (crossval for params, best for choosing a model)')
    args = parser.parse_args()

    data = load_all_data(args.root_dir, filt=is_teleop)

    data_filt_clipped = [d for d in data if d['goal_success'] and d['morsel_mask'][d['morsel_target']] and d['fix_data'] is not None]
    for d in data_filt_clipped:
        d['fix_data'] = d['fix_data'].loc[d['fix_data'].start_timestamp >= 0., :] # fix for harmonic 0.5.0

    seqs = data_to_seqs(data_filt_clipped)
    default_remap = np.array([0,1,1,1,1,1,1,1,2,2,3,4,5])
    default_goal_remaps = [ np.arange(6), np.arange(6), np.arange(6) ]
    for i in range(len(default_goal_remaps)):
        default_goal_remaps[i][default_goal_remaps[i] >= 3] = 4 # remap incorrect labels -> 4
        default_goal_remaps[i][3+i] = 3 # remap correct -> 3


    evals = []
    if args.method == 'crossval':
        score_fn = eval_crossval
    elif args.method == 'best':
        score_fn = eval_model
    else:
        assert False


    widgets =  [ progressbar.widgets.Percentage(),
                ' ', progressbar.widgets.SimpleProgress(),
                ' (', progressbar.widgets.FormatLabel('n_comp: {variables.n_comp}, nmax: {variables.nmax}, dt: {variables.dt}', new_style=True), ')',
                ' ', progressbar.widgets.Bar(),
                ' ', progressbar.widgets.Timer(),
                ' ', progressbar.widgets.AdaptiveETA() ]
    bar = progressbar.bar.ProgressBar(max_value=len(args.n_components)*len(args.nmax)*len(args.dt), widgets=widgets, variables={'n_comp': None, 'nmax': None, 'dt': None}).start()

    for ct, (n_comp, nmax, dt) in enumerate(itertools.product(args.n_components, args.nmax, args.dt)):
        bar.update(ct, n_comp=n_comp, nmax=nmax, dt=dt)
        processor = SequenceProcessor(label_remap=default_remap, goal_remaps=default_goal_remaps, dt=dt, nmax=nmax)
        model = hmmlearn.hmm.MultinomialHMM(n_components=n_comp)
        evals.append({'n_comp': n_comp, 'nmax': nmax, 'dt': dt, 
            **score_fn(model, processor, seqs)})
    bar.finish()

    all_evals = pd.DataFrame(evals)
    print(all_evals)
    all_evals.to_csv('hmm_evals.csv')

    # print('startprob: {}'.format(model.startprob_))
    # print('transmat: {}'.format(model.transmat_))
    # print('emissionprob: {}'.format(model.emissionprob_))



if __name__ == "__main__":
    main()
    
