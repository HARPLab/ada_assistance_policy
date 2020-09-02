# The basic ADA interface, such as executing end effector motions, getting the state of the robot, etc.
from collections import namedtuple
from AdaAssistancePolicy import *
from UserBot import *
import AssistancePolicyVisualizationTools as vistools
# from GoalPredictor import *

from DataRecordingUtils import TrajectoryData

import numpy as np
import math
import time
import os
import sys
import cPickle as pickle
import argparse

import tf
import tf.transformations as transmethods
import rospkg
import rospy
# import roslib

import openravepy
import adapy
import prpy

from ada_teleoperation.AdaTeleopHandler import AdaTeleopHandler, Is_Done_Func_Button_Hold
from ada_teleoperation.RobotState import *
from GoalPredictor import PolicyBasedGoalPredictor, MergedGoalPredictor
from GazeBasedPredictor import load_gaze_predictor_from_params


CONTROL_HZ = 40.


ROBOT_OPTS = ['env', 'robot']
GOAL_OPTS = ['goals', 'goal_objects', 'goal_object_poses']
INPUT_OPTS = ['input_interface_name', 'num_input_dofs',
              'use_finger_mode']


class AdaHandlerConfig(namedtuple('AdaHandlerConfig', ROBOT_OPTS + GOAL_OPTS + INPUT_OPTS)):

    @classmethod
    def create(cls, *args, **kwargs):
        # super() constructer doesn't support missing params
        # so fill them in before
        # first remap positional to keyword args
        kwargs.update({k: v for k, v in zip(cls._fields, args)})

        if 'goal_object_poses' not in kwargs:
            kwargs['goal_object_poses'] = [ goal_obj.GetTransform()
                                           for goal_obj in kwargs['goal_objects'] ]

        config = cls(**kwargs)

        # Validation of various stuff goes here!
        config._cache = {}

        return config

    def get_user_input(self, sim=False):
        if sim:
            return UserBot(self.goals, self.robot, self.get_teleop_handler().robot_state)
        else:
            return self.get_teleop_handler()

    def get_teleop_handler(self):
        # cache this so we can return a copy for user input if appropriate
        if 'teleop_handler' not in self._cache:
            self._cache['teleop_handler'] = AdaTeleopHandler(
                self.env, self.robot, self.input_interface_name, self.num_input_dofs, self.use_finger_mode)
        return self._cache['teleop_handler']




POLICY_OPTS = ['direct_teleop_only',
               'blend_only', 'fix_magnitude_user_command', 'transition_function']
TRIAL_OPTS = ['simulate_user', 'is_done_func', 'finish_trial_func']
OUTPUT_OPTS = ['traj_data_recording']
PREDICTION_OPTS = ['predict_policy', 'predict_gaze']


class PolicyConfig(namedtuple('PolicyConfig', POLICY_OPTS + TRIAL_OPTS + OUTPUT_OPTS + PREDICTION_OPTS)):
    __DEFAULT_OPTS__ = {
        'direct_teleop_only': False,
        'blend_only': False,
        'fix_magnitude_user_command': False,
        'is_done_func': Is_Done_Func_Button_Hold,
        'transition_function': lambda x, y: x+y,
        'simulate_user': False,
        'finish_trial_func': None,
        'traj_data_recording': None,
        'predict_policy': True,
        'predict_gaze': False
    }

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs.update({k: v for k, v in zip(cls._fields, args)})
        vals = cls.__DEFAULT_OPTS__.copy()
        vals.update(kwargs)

        config = cls(**vals)

        config._cache = {}

        return config

    def get_goal_predictor(self, goals):
        predictors = []
        if self.predict_policy:
            predictors.append(GoalPredictor(self.get_rl_policy(goals)))
        if self.predict_gaze:
            predictors.append(load_gaze_predictor_from_params())
        
        if len(predictors) > 1:
            return MergedGoalPredictor(predictors) # todo: weights
        elif len(predictors) == 1:
            return predictors[0]
        else:
            raise RuntimeError("no prediction strategy specified!")

    def get_rl_policy(self, goals):
        # cache for repeated access to same object
        # for probability creation
        if 'rl_policy' not in self._cache:
            rl_policy = AssistancePolicy(goals)
            if self.fix_magnitude_user_command:
                # update the constants appropriately
                for goal_policy in rl_policy.goal_assist_policies:
                    for target_policy in goal_policy.target_assist_policies:
                        target_policy.set_constants(
                            **FixMagnitudeSharedAutonPolicy.__HUBER_CONSTS__)
            self._cache['rl_policy'] = rl_policy
        return self._cache['rl_policy']

    def get_assistance_policy(self):
        if self.direct_teleop_only:
            return DirectTeleopPolicy()
        elif self.blend_only:
            return BlendPolicy()
        elif self.fix_magnitude_user_command:
            return FixMagnitudeSharedAutonPolicy(self.transition_function)
        else:
            return SharedAutonPolicy(self.transition_function)


class AdaHandler:
    def __init__(self, config):
        self.config = config
        self.ada_teleop = config.get_teleop_handler()
        self.user_input_mapper = self.ada_teleop.user_input_mapper

    def execute_policy(self, policy_config):

        # TODO: move this to config
        time_per_iter = 1./CONTROL_HZ

        # load in the objects for executing the policy
        user_input = self.config.get_user_input(policy_config.simulate_user)
        rl_policy = policy_config.get_rl_policy(self.config.goals)
        goal_predictor = policy_config.get_goal_predictor(self.config.goals)
        assistance_policy = policy_config.get_assistance_policy(self.robot_policy)

        robot_state = self.ada_teleop.robot_state

        # if specified traj data for recording, initialize
        traj_data_recording = policy_config.traj_data_recording
        if traj_data_recording is not None:
            traj_data_recording.set_init_info(
                start_state=copy.deepcopy(robot_state),
                goals=copy.deepcopy(self.config.goals),
                input_interface_name=self.ada_teleop.teleop_interface,
                assist_type=policy.assist_type)
        vis = vistools.VisualizationHandler()

        def _do_update(evt):
            # update the robot state to match the actual robot position
            robot_state.ee_trans = self.config.robot.arm.GetEndEffectorTransform()
            ee_trans = robot_state.ee_trans

            # get the user input
            try:
                user_input_all = user_input.get_user_command()
                direct_teleop_action = self.user_input_mapper.input_to_action(
                    user_input_all, robot_state)
            except RuntimeError as e:
                rospy.logwarn('Failed to get any input info: {}'.format(e))
                user_input_all = None
                direct_teleop_action = Action()                


            # update the policy
            rl_policy.update(robot_state, direct_teleop_action)
            # get the goal probabilities
            # must be AFTER rl_policy.update()
            # since if we're using policy-based prediction, we're using the SAME policy object
            # both here to generate candidate actions and in the goal predictor
            goal_distribution = goal_predictor.get_distribution()

            # if left trigger is being hit (i.e. mode switch), bypass assistance
            if user_input_all is not None and user_input_all.button_changes[0] == 1:
                action = direct_teleop_action
            else:
                # defer to the assistance policy object to determine the appropriate fused action
                action = policy.get_action(
                    robot_state, direct_teleop_action, config.goals, goal_distribution, rl_policy.get_robot_actions())

            # execute robot action
            self.ada_teleop.ExecuteAction(action)    

            ### visualization ###
            vis.draw_probability_text(
                self.config.goal_object_poses, self.robot_policy.goal_predictor.get_distribution())

            vis.draw_action_arrows(
                ee_trans, direct_teleop_action.twist[0:3], action.twist[0:3]-direct_teleop_action.twist[0:3])

            ### logging ###
            if traj_data_recording:
                robot_dof_values = self.config.robot.GetDOFValues()
                traj_data_recording.add_datapoint(
                    robot_state = copy.deepcopy(robot_state), 
                    robot_dof_values = copy.copy(robot_dof_values), 
                    user_input_all = copy.deepcopy(user_input_all), 
                    direct_teleop_action = copy.deepcopy(direct_teleop_action), 
                    executed_action = copy.deepcopy(action), 
                    goal_distribution = goal_distribution)

            ### check if we're still running or not ###
            if policy_config.is_done_func(self.config.env, self.config.robot, user_input_all) or rospy.is_shutdown():
                self._is_running = False
                self.timer.shutdown()

        # set up timer callback
        self._is_running = True
        self.timer = rospy.Timer(rospy.Duration(
            time_per_iter), _do_update)
        # spin till the callback ends
        while self._is_running and not rospy.is_shutdown():
            rospy.sleep(time_per_iter)

        # execute zero velocity to stop movement
        self.ada_teleop.execute_joint_velocities(
            np.zeros(len(self.config.robot.arm.GetDOFValues())))

        # set the intended goal and write data to file
        if traj_data_recording:
            values, qvalues = self.robot_policy.assist_policy.get_values()
            traj_data_recording.set_end_info(
                intended_goal_ind=np.argmin(values))
            traj_data_recording.tofile()

        if policy_config.finish_trial_func is not None:
            policy_config.finish_trial_func()
