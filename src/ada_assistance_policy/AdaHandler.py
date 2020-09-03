import numpy as np
import tf
import rospy
from collections import namedtuple

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


CONTROL_HZ = 40.


GOAL_OPTS = ['goals', 'goal_objects', 'goal_object_poses']
INPUT_OPTS = ['input_interface_name', 'num_input_dofs',
              'use_finger_mode']


class AdaHandlerConfig(namedtuple('AdaHandlerConfig', GOAL_OPTS + INPUT_OPTS)):

    @classmethod
    def create(cls, *args, **kwargs):
        # super() constructer doesn't support missing params
        # so fill them in before
        # first remap positional to keyword args
        kwargs.update({k: v for k, v in zip(cls._fields, args)})

        if 'goal_object_poses' not in kwargs:
            kwargs['goal_object_poses'] = [goal_obj.GetTransform()
                                           for goal_obj in kwargs['goal_objects']]

        config = cls(**kwargs)

        # Validation of various stuff goes here!
        return config

    def get_teleop_handler(self, env, robot):
        return AdaTeleopHandler(
            env, robot, self.input_interface_name, self.num_input_dofs, self.use_finger_mode)


POLICY_OPTS = ['direct_teleop_only',
               'blend_only', 'fix_magnitude_user_command', 'transition_function']
TRIAL_OPTS = ['simulate_user', 'is_done_func']
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

        return config

    def get_goal_predictor(self, goals, rl_policy=None):
        predictors = []
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
        return get_assistance_policy(direct_teleop_only=self.direct_teleop_only,
                                     blend_only=self.blend_only, fix_magnitude_user_command=self.fix_magnitude_user_command,
                                     transition_function=self.transition_function)

    def get_user_input(self, teleop_handler, robot, goals):
        if self.simulate_user:
            return UserBot(goals, robot, teleop_handler.robot_state)
        else:
            return teleop_handler


class AdaHandlerFuture(Future):
    def __init__(self, env, robot, config, policy_config):
        super(AdaHandlerFuture, self).__init__()

        # load in the variables from the configs
        self.config = config
        self.env = env
        self.robot = robot
        self.goals = config.goals
        self.ada_teleop = config.get_teleop_handler(env, robot)
        self.user_input_mapper = self.ada_teleop.user_input_mapper

        self.user_input = policy_config.get_user_input(self.ada_teleop,
                                                  self.robot, self.goals)
        self.rl_policy = policy_config.get_rl_policy(self.goals)
        self.goal_predictor = policy_config.get_goal_predictor(
            self.goals, self.rl_policy)
        self.assistance_policy = policy_config.get_assistance_policy()

        self.traj_data_recording = policy_config.traj_data_recording
        if self.traj_data_recording is not None:
            traj_data_recording.set_init_info(
                start_state=copy.deepcopy(self.ada_teleop.robot_state),
                goals=copy.deepcopy(self.goals),
                input_interface_name=self.ada_teleop.teleop_interface,
                assist_type=self.assistance_policy.assist_type)
        self.vis = vistools.VisualizationHandler()
        self.is_done_func = policy_config.is_done_func

        # start the callback thread
        self._cancel_requested = False
        self.timer = rospy.Timer(rospy.Duration(
            1./CONTROL_HZ), self._do_update)

    def cancel(self):
        self._cancel_requested = True

    def _do_update(self, evt):
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
            rospy.logwarn('Failed to get any input info: {}'.format(e))
            user_input_all = None
            direct_teleop_action = Action()

        # update the policy
        self.rl_policy.update(robot_state, direct_teleop_action)
        # get the goal probabilities
        # must be AFTER rl_policy.update()
        # since if we're using policy-based prediction, we're using the SAME policy object
        # both here to generate candidate actions and in the goal predictor
        goal_distribution = self.goal_predictor.get_distribution()

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
            self.config.goal_object_poses, goal_distribution)

        self.vis.draw_action_arrows(
            robot_state.ee_trans, direct_teleop_action.twist[0:3], action.twist[0:3]-direct_teleop_action.twist[0:3])

        ### logging ###
        if self.traj_data_recording:
            robot_dof_values = self.robot.GetDOFValues()
            self.traj_data_recording.add_datapoint(
                robot_state=copy.deepcopy(robot_state),
                robot_dof_values=copy.copy(robot_dof_values),
                user_input_all=copy.deepcopy(user_input_all),
                direct_teleop_action=copy.deepcopy(direct_teleop_action),
                executed_action=copy.deepcopy(action),
                goal_distribution=goal_distribution)

        ### check if we're still running or not ###
        if self._cancel_requested or self.is_done_func(self.env, self.robot, user_input_all):
            self._finalize()
            if self._cancel_requested:
                self.set_cancelled()
            else:
                self.set_result(None)
    
    def _finalize(self):
        # stop the timer
        self.timer.shutdown()
        # execute zero velocity to stop movement
        self.ada_teleop.execute_joint_velocities(
            np.zeros(len(self.robot.arm.GetDOFValues())))

        # set the intended goal and write data to file
        if self.traj_data_recording is not None:
            values, qvalues = self.rl_policy.get_values()
            self.traj_data_recording.set_end_info(
                intended_goal_ind=np.argmin(values))
            self.traj_data_recording.tofile()

class AdaHandler:
    def __init__(self, env, robot, config):
        self.config = config
        self.env = env
        self.robot = robot
        self.goals = config.goals
        self.ada_teleop = config.get_teleop_handler(env, robot)
        self.user_input_mapper = self.ada_teleop.user_input_mapper

    def execute_policy(self, policy_config):

        # TODO: move this to config
        time_per_iter = 1./CONTROL_HZ

        # load in the objects for executing the policy
        user_input = policy_config.get_user_input(self.ada_teleop,
                                                  self.robot, self.goals)
        rl_policy = policy_config.get_rl_policy(self.goals)
        goal_predictor = policy_config.get_goal_predictor(
            self.goals, rl_policy)
        assistance_policy = policy_config.get_assistance_policy()

        robot_state = self.ada_teleop.robot_state

        # if specified traj data for recording, initialize
        traj_data_recording = policy_config.traj_data_recording
        if traj_data_recording is not None:
            traj_data_recording.set_init_info(
                start_state=copy.deepcopy(robot_state),
                goals=copy.deepcopy(self.goals),
                input_interface_name=self.ada_teleop.teleop_interface,
                assist_type=policy.assist_type)
        vis = vistools.VisualizationHandler()

        def _do_update(evt):
            # update the robot state to match the actual robot position
            # TODO: move this somewhere less terrible
            robot_state.ee_trans = self.robot.arm.GetEndEffectorTransform()

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
                action = assistance_policy.get_action(
                    robot_state, direct_teleop_action, self.goals, goal_distribution, rl_policy.get_robot_actions())

            # execute robot action
            self.ada_teleop.ExecuteAction(action)

            ### visualization ###
            vis.draw_probability_text(
                self.config.goal_object_poses, goal_distribution)

            vis.draw_action_arrows(
                robot_state.ee_trans, direct_teleop_action.twist[0:3], action.twist[0:3]-direct_teleop_action.twist[0:3])

            ### logging ###
            if traj_data_recording:
                robot_dof_values = self.robot.GetDOFValues()
                traj_data_recording.add_datapoint(
                    robot_state=copy.deepcopy(robot_state),
                    robot_dof_values=copy.copy(robot_dof_values),
                    user_input_all=copy.deepcopy(user_input_all),
                    direct_teleop_action=copy.deepcopy(direct_teleop_action),
                    executed_action=copy.deepcopy(action),
                    goal_distribution=goal_distribution)

            ### check if we're still running or not ###
            if policy_config.is_done_func(self.env, self.robot, user_input_all) or rospy.is_shutdown():
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
            np.zeros(len(self.robot.arm.GetDOFValues())))

        # set the intended goal and write data to file
        if traj_data_recording:
            values, qvalues = rl_policy.get_values()
            traj_data_recording.set_end_info(
                intended_goal_ind=np.argmin(values))
            traj_data_recording.tofile()

        if policy_config.finish_trial_func is not None:
            policy_config.finish_trial_func()
