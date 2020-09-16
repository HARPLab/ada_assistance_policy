import csv
import logging
import numpy as np
import os
import tf
import rospy
from collections import namedtuple, OrderedDict
import traceback
import yaml

from adapy.futures import Future
from ada_teleoperation.AdaTeleopHandler import AdaTeleopHandler, Is_Done_Func_Button_Hold
from ada_teleoperation.RobotState import Action
from GoalPredictor import PolicyBasedGoalPredictor, MergedGoalPredictor
from GazeBasedPredictor import load_gaze_predictor_from_params
from AdaAssistancePolicy import get_assistance_policy
from UserBot import UserBot
import AssistancePolicyVisualizationTools as vistools
from DataRecordingUtils import TrajectoryData
from AssistancePolicy import AssistancePolicy

logger = logging.getLogger('ada_assistance_policy')

CONTROL_HZ = 20. # DROPPED FROM 40 SO GUI CAN GET A CHANCE TO DO THINGS


GOAL_OPTS = ['goals']
INPUT_OPTS = ['input_interface_name', 'num_input_dofs',
              'use_finger_mode']
POLICY_OPTS = ['direct_teleop_only',
               'blend_only', 'fix_magnitude_user_command', 'transition_function']
TRIAL_OPTS = ['simulate_user']
OUTPUT_OPTS = ['log_dir']
PREDICTION_OPTS = ['predict_policy', 'predict_gaze', 'pick_goal']

class AdaHandlerConfig(namedtuple('AdaHandlerConfig', GOAL_OPTS + INPUT_OPTS + POLICY_OPTS + TRIAL_OPTS + OUTPUT_OPTS + PREDICTION_OPTS)):
    __DEFAULT_OPTS__ = {
        'input_interface_name': 'kinova',
        'num_input_dofs': 2,
        'use_finger_mode': False,
        'direct_teleop_only': False,
        'blend_only': False,
        'fix_magnitude_user_command': False,
        'transition_function': lambda x, y: x+y,
        'simulate_user': False,
        'log_dir': None,
        'predict_policy': True,
        'predict_gaze': False,
        'pick_goal': False
    }

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs.update({k: v for k, v in zip(cls._fields, args)})
        vals = cls.__DEFAULT_OPTS__.copy()
        vals.update(kwargs)

        config = cls(**vals)
        config._random_goal_index = None

        return config

    def get_random_goal_index(self):
        if self._random_goal_index is None:
            self._random_goal_index = np.random.randint(len(self.goals))
        return self._random_goal_index

    def get_is_done_fn(self):
        if self.pick_goal:
            # we want to auto stop when we're at the goal
            goal = self.goals[self.get_random_goal_index()]
            def is_done_at_goal(env, robot, input):
                return goal.at_goal(robot.GetActiveManipulator().GetEndEffectorTransform())
            return is_done_at_goal
        else:
            return Is_Done_Func_Button_Hold

    def get_teleop_handler(self, env, robot):
        return AdaTeleopHandler(
            env, robot, self.input_interface_name, self.num_input_dofs, self.use_finger_mode)

    def get_goal_predictor(self, goals, rl_policy=None):
        predictors = []
        if self.pick_goal:
            # simulate a random goal
            prob = np.zeros((len(goals),))
            if len(goals) > 0: 
                prob[self.get_random_goal_index()] = 1.
            # build a "generator" that just returns that goal
            class RandomPick:
                def get_distribution(self):
                    return prob
            return RandomPick()

        # otherwise make an actual predictor
        if self.predict_policy:
            if rl_policy is None:
                raise ValueError(
                    'PolicyBasedGoalPredictor requested but no rl_policy provided')
            predictors.append(PolicyBasedGoalPredictor(rl_policy))
        if self.predict_gaze:
            predictors.append(load_gaze_predictor_from_params())

        if len(predictors) > 1:
            return MergedGoalPredictor(predictors)  # todo: weights
        elif len(predictors) == 1:
            return predictors[0]
        else:
            raise RuntimeError("no prediction strategy specified!")

    def get_rl_policy(self, goals):
        rl_policy = AssistancePolicy(goals)
        if self.fix_magnitude_user_command:
            # update the constants appropriately
            for goal_policy in rl_policy.goal_assist_policies:
                for target_policy in goal_policy.target_assist_policies:
                    target_policy.set_constants(
                        **FixMagnitudeSharedAutonPolicy.__HUBER_CONSTS__)
        return rl_policy

    def get_assistance_policy(self):
        if not self.goals and not self.direct_teleop_only:
            raise ValueError('Requested assistance but no goals detected!')
        return get_assistance_policy(direct_teleop_only=self.direct_teleop_only,
                                     blend_only=self.blend_only, fix_magnitude_user_command=self.fix_magnitude_user_command,
                                     transition_function=self.transition_function)

    def get_user_input(self, teleop_handler, robot, goals):
        if self.simulate_user:
            return UserBot(goals, robot, teleop_handler.robot_state)
        else:
            return teleop_handler


def _array_to_str(arr):
    return np.array2string(arr, precision=6).replace('\n', '')

class AdaHandler(Future):
    def __init__(self, env, robot, config, loggers): # TODO(rma): clean up multiple configs (probably by moving ada_meal_scenario.assistance.assistance_config into this package)
        super(AdaHandler, self).__init__()

        # load in the variables from the configs
        self.config = config
        self.env = env
        self.robot = robot
        self.goals = config.goals
        self.ada_teleop = config.get_teleop_handler(env, robot)
        self.user_input_mapper = self.ada_teleop.user_input_mapper

        self.user_input = config.get_user_input(self.ada_teleop,
                                                  self.robot, self.goals)
        self.rl_policy = config.get_rl_policy(self.goals)
        self.assistance_policy = config.get_assistance_policy()
        self.goal_predictor = config.get_goal_predictor(
            self.goals, self.rl_policy)

        # logging variables
        self.loggers = loggers
        for logger in self.loggers:
            logger.start()

        # log the initial data
        if config.log_dir is not None:
            data = { 
                'goals': {g.name: g.pose.tolist() for g in self.goals },
                'assistance_type': self.assistance_policy.assist_type,
                'goal_predictor': self.goal_predictor.get_config()
            }
            with open(os.path.join(config.log_dir, 'assistance_init.yaml'), 'w') as f:
                yaml.safe_dump(data, f)

            self._log_file = open(os.path.join(config.log_dir, 'assistance_data.csv'), 'w')
            self._logger = None  # lazily init so we can get the field names
        else:
            self._log_file = None

        self.vis = vistools.VisualizationHandler()
        self.is_done_func = config.get_is_done_fn()

        # start the callback thread
        self._cancel_requested = False
        self.timer = rospy.Timer(rospy.Duration(
            1./CONTROL_HZ), self._do_update)

    def cancel(self):
        logger.debug('cancel requested')
        self._cancel_requested = True

    def _do_update(self, evt):
        # see if we were cancelled before we do anything
        if self._cancel_requested:
            self._finalize()
            self.set_cancelled()

        try:
            # update the robot state to match the actual robot position
            # TODO: move this somewhere less terrible
            robot_state = self.ada_teleop.robot_state
            robot_state.ee_trans = self.robot.arm.GetEndEffectorTransform()

            # get the user input
            try:
                user_input_all = self.user_input.get_user_command()
                direct_teleop_action = self.user_input_mapper.input_to_action(
                    user_input_all, robot_state)
            except RuntimeError as e:
                rospy.logwarn_throttle(5, 'Failed to get input info: {}'.format(e))
                user_input_all = None
                direct_teleop_action = Action()

            # update the policy
            self.rl_policy.update(robot_state, direct_teleop_action)
            # get the goal probabilities
            # must be AFTER rl_policy.update()
            # since if we're using policy-based prediction, we're using the SAME policy object
            # both here to generate candidate actions and in the goal predictor
            # 
            # Also, make sure to get the log in the SAME function call
            # so that we are sure the data doesn't change between getting the dist and getting the log
            # (e.g. a new gaze data point comes in on a different thread)
            goal_distribution, goal_log = self.goal_predictor.get_distribution(get_log=True)

            # if left trigger is being hit (i.e. mode switch), bypass assistance
            if user_input_all is not None and user_input_all.button_changes[0] == 1:
                action = direct_teleop_action
            else:
                # defer to the assistance policy object to determine the appropriate fused action
                action = self.assistance_policy.get_action(
                    robot_state, direct_teleop_action, self.goals, goal_distribution, self.rl_policy.get_robot_actions())

            # execute robot action
            self.ada_teleop.ExecuteAction(action)

            ### visualization ###
            self.vis.draw_probability_text(
                [ goal.overlay_pose for goal in self.goals ], goal_distribution)

            self.vis.draw_action_arrows(
                robot_state.ee_trans, direct_teleop_action.twist[0:3], action.twist[0:3]-direct_teleop_action.twist[0:3])

            ### logging ###
            if self._log_file is not None:
                # collect the data we want to log
                # TODO: is it too slow to log directly to disk, or does autobuffering work well enough? Could store all
                # of this and save to disk in finalize(), but then (1) [tiny] memory usage (2) if it crashes we lose partial data
                data = OrderedDict()
                data['timestamp'] = evt.current_real.to_sec()
                data['robot_dofs'] = _array_to_str(self.robot.GetDOFValues())
                data['robot_ee_trans'] = _array_to_str(robot_state.ee_trans)
                data.update({ 'input_{}'.format(k):v for k,v in user_input_all.as_dict().items() })
                data.update({ 'direct_teleop_{}'.format(k):v for k,v in direct_teleop_action.as_dict().items() })
                data.update({ 'p_goal_{}'.format(i): _array_to_str(v) for i, v in enumerate(goal_distribution)})
                data.update({ 'goal_{}'.format(k): v for k, v in goal_log.items() })
                data.update({ 'executed_action_{}'.format(k):v for k,v in action.as_dict().items() })

                # see if we need to initialize the logger
                if self._logger is None:
                    self._logger = csv.DictWriter(self._log_file, data.keys())
                    self._logger.writeheader()
                
                self._logger.writerow(data)

            ### check if we're still running or not ###
            if self._cancel_requested or self.is_done_func(self.env, self.robot, user_input_all):
                self._finalize()
                if self._cancel_requested:
                    self.set_cancelled()
                else:
                    self.set_result(None)
        except Exception as e:
            traceback.print_exc()
            # make sure to clean up
            self._finalize()
            # transfer the exception to the main thread
            self.set_exception(e)
    
    def _finalize(self):
        # stop the timer
        self.timer.shutdown()
        # execute zero velocity to stop movement
        self.ada_teleop.execute_joint_velocities(
            np.zeros(len(self.robot.arm.GetDOFValues())))

        # stop all loggers
        for logger in self.loggers:
            logger.stop()
       
        # clean up internal logger
        if self._log_file is not None:
            self._log_file.close()