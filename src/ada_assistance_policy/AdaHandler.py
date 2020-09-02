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


class PolicyConfig(namedtuple('PolicyConfig', POLICY_OPTS + TRIAL_OPTS + OUTPUT_OPTS)):
    __DEFAULT_OPTS__ = {
        'direct_teleop_only': False,
        'blend_only': False,
        'fix_magnitude_user_command': False,
        'is_done_func': Is_Done_Func_Button_Hold,
        'transition_function': lambda x, y: x+y,
        'simulate_user': False,
        'finish_trial_func': None,
        'traj_data_recording': None
    }

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs.update({k: v for k, v in zip(cls._fields, args)})
        vals = cls.__DEFAULT_OPTS__.copy()
        vals.update(kwargs)

        config = cls(**vals)

        return config

    def get_assistance_policy(self, robot_policy):
        if self.direct_teleop_only:
            return DirectTeleopPolicy()
        elif self.blend_only:
            return BlendPolicy(robot_policy)
        elif self.fix_magnitude_user_command:
            return FixMagnitudeSharedAutonPolicy(robot_policy, self.transition_function)
        else:
            return SharedAutonPolicy(robot_policy, self.transition_function)


class DirectTeleopPolicy:
    def __init__(self, *args, **kwargs):
        self.assist_type = 'None'

    def get_action(self, direct_action):
        return direct_action


class BlendPolicy:
    def __init__(self, robot_policy, *args, **kwargs):
        self.assist_type = 'blend'
        self.robot_policy = robot_policy

    def get_action(self, direct_action):
        return self.robot_policy.get_blend_action()


class SharedAutonPolicy:
    def __init__(self, robot_policy, transition_function, *args, **kwargs):
        self.assist_type = 'shared_auton'
        self.robot_policy = robot_policy
        self.transition_function = transition_function

    def get_action(self, direct_action):
        return self.robot_policy.get_action(fix_magnitude_user_command=False,
                                            transition_function=self.transition_function)


class FixMagnitudeSharedAutonPolicy:
    __HUBER_CONSTS__ = {
        'huber_translation_linear_multiplier': 1.55,
        'huber_translation_delta_switch': 0.11,
        'huber_translation_constant_add': 0.2,
        'huber_rotation_linear_multiplier': 0.20,
        'huber_rotation_delta_switch': np.pi/72.,
        'huber_rotation_constant_add': 0.3,
        'huber_rotation_multiplier': 0.20,
        'robot_translation_cost_multiplier': 14.0,
        'robot_rotation_cost_multiplier': 0.05
    }

    def __init__(self, robot_policy, transition_function, *args, **kwargs):
        self.assist_type = 'shared_auton_prop'
        self.robot_policy = robot_policy
        self.transition_function = transition_function

        # update the constants appropriately
        for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
            for target_policy in goal_policy.target_assist_policies:
                target_policy.set_constants(
                    **FixMagnitudeSharedAutonPolicy.__HUBER_CONSTS__)

    def get_action(self, direct_action):
        return self.robot_policy.get_action(fix_magnitude_user_command=True,
                                            transition_function=self.transition_function)


class AdaHandler:
    def __init__(self, config):
        self.config = config
        self.robot_policy = AdaAssistancePolicy(config.goals)
        self.ada_teleop = config.get_teleop_handler()
        self.user_input_mapper = self.ada_teleop.user_input_mapper

    def execute_policy(self, policy_config):

        time_per_iter = 1./CONTROL_HZ

        user_input = self.config.get_user_input(policy_config.simulate_user)
        policy = policy_config.get_assistance_policy(self.robot_policy)

        robot_state = self.ada_teleop.robot_state
        robot_state.ee_trans = self.config.robot.arm.GetEndEffectorTransform()
        ee_trans = robot_state.ee_trans
        robot_dof_values = self.config.robot.GetDOFValues()

        # if specified traj data for recording, initialize
        traj_data_recording = policy_config.traj_data_recording
        if traj_data_recording:

            traj_data_recording.set_init_info(start_state=copy.deepcopy(robot_state),
                                              goals=copy.deepcopy(
                                                  self.config.goals),
                                              input_interface_name=self.ada_teleop.teleop_interface,
                                              assist_type=policy.assist_type)
        vis = vistools.VisualizationHandler()

        def _do_update(evt):
            try:
                user_input_all = user_input.get_user_command()
                direct_teleop_action = self.user_input_mapper.input_to_action(
                    user_input_all, robot_state)
            except RuntimeError as e:
                rospy.logwarn('Failed to get any input info: {}'.format(e))
                user_input_all = None
                direct_teleop_action = Action()                


            # update the policy
            self.robot_policy.update(robot_state, direct_teleop_action)
            # if left trigger is being hit (i.e. mode switch), bypass assistance
            if user_input_all is not None and user_input_all.button_changes[0] == 1:
                action = direct_teleop_action
            else:
                action = policy.get_action(direct_teleop_action)

            self.ada_teleop.ExecuteAction(action)    # execute robot action

            ### visualization ###
            vis.draw_probability_text(
                self.config.goal_object_poses, self.robot_policy.goal_predictor.get_distribution())

            vis.draw_action_arrows(
                ee_trans, direct_teleop_action.twist[0:3], action.twist[0:3]-direct_teleop_action.twist[0:3])

            ### end visualization ###

            if traj_data_recording:
                traj_data_recording.add_datapoint(robot_state=copy.deepcopy(robot_state), robot_dof_values=copy.copy(robot_dof_values), user_input_all=copy.deepcopy(
                    user_input_all), direct_teleop_action=copy.deepcopy(direct_teleop_action), executed_action=copy.deepcopy(action), goal_distribution=self.robot_policy.goal_predictor.get_distribution())

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
