#Handles converting openrave items to generic assistance policy
from Goal import *
import GoalPredictor as GoalPredictor
from ada_teleoperation.RobotState import Action
from AssistancePolicy import *
from OpenraveUtils import *
import math
import time
# import GazeBasedPredictor

ADD_MORE_IK_SOLS = False


class DirectTeleopPolicy:
    def __init__(self, *args, **kwargs):
        self.assist_type = 'None'

    def get_action(self, robot_state, direct_action, goals, goal_distribution, twist_candidates):
        return direct_action


class BlendPolicy:
    def __init__(self, *args, **kwargs):
        self.assist_type = 'blend'
        # TODO: set from config?
        self.check_confidence = BlendPolicy.blend_confidence_function_prob_diff 

    def get_action(self, robot_state, direct_action, goals, goal_distribution, twist_candidates):
        max_prob_goal_ind = np.argmax(goal_distribution)

        #check if we meet the confidence criteria which dictates whether or not assistance is provided
        #use the one from ancas paper - euclidean distance and some threshhold
        #if blend_confidence_function_euclidean_distance(self.robot_state, self.goals[max_prob_goal_ind]):
        if self.check_confidence(robot_state, goals[max_prob_goal_ind], goal_distribution):
            return Action(twist=twist_candidates[max_prob_goal_ind],
                        finger_vel=direct_action.finger_vel,
                        switch_mode_to=direct_action.switch_mode_to)

        else:
            #if we don't meet confidence function, use direct teleop
            return direct_action

    @staticmethod
    def blend_confidence_function_prob_diff(robot_state, goal, goal_distribution, prob_diff_required=0.4):
        if len(goal_distribution) <= 1:
            return True

        goal_distribution_sorted = np.sort(goal_distribution)
        return goal_distribution_sorted[-1] - goal_distribution_sorted[-2] > prob_diff_required

    @staticmethod
    def blend_confidence_function_euclidean_distance(robot_state, goal, goal_distribution, distance_thresh=0.10):
        manip_pos = robot_state.get_pos()
        goal_poses = goal.target_poses
        goal_pose_distances = [np.linalg.norm(manip_pos - pose[0:3,3]) for pose in goal_poses]
        dist_to_nearest_goal = np.min(goal_pose_distances)
        return dist_to_nearest_goal < distance_thresh


class SharedAutonPolicy:
    def __init__(self, transition_function, *args, **kwargs):
        self.assist_type = 'shared_auton'
        self.transition_function = transition_function

    def get_action(self, robot_state, direct_action, goals, goal_distribution, twist_candidates):
        #TODO how do we handle mode switch vs. not?
        total_action_twist = np.zeros(GoalPolicy.TargetPolicy.ACTION_DIMENSION)

        # take the expected robot action over the candidates
        for goal_action, goal_prob in zip(action_candidates, goal_distribution):
            total_action_twist += goal_prob * goal_action
        total_action_twist /= np.sum(goal_distribution)

        to_ret_twist = transition_function(
            total_action_twist, direct_action.twist)  # a + u from paper

        return Action(twist=to_ret_twist,
                    finger_vel=direct_action.finger_vel,
                    switch_mode_to=direct_action.switch_mode_to)

class FixMagnitudeSharedAutonPolicy(SharedAutonPolicy):
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
        super(FixMagnitudeSharedAutonPolicy, self).__init__(robot_policy, transition_function, *args, **kwargs)
        self.assist_type = 'shared_auton_prop'

    def get_action(self, robot_state, direct_action, goals, goal_distribution, twist_candidates):
        action = super(FixMagnitudeSharedAutonPolicy, self).get_action(
            self, robot_state, direct_action, goals, goal_distribution, twist_candidates)
        # fix the magnitude as requested
        action.twist *= np.linalg.norm(direct_action.twist) / np.linalg.norm(action.twist)
        return action


def get_assistance_policy(direct_teleop_only, blend_only, fix_magnitude_user_command, transition_function):
    if direct_teleop_only:
        return DirectTeleopPolicy()
    elif blend_only:
        return BlendPolicy()
    elif fix_magnitude_user_command:
        return FixMagnitudeSharedAutonPolicy(transition_function)
    else:
        return SharedAutonPolicy(transition_function)

#generic functions
def goal_from_object(object, manip):
  pose = object.GetTransform()
  robot = manip.GetRobot()
  env = robot.GetEnv()

  #generate TSRs for object
  if not 'bowl' in object.GetName():
    target_tsrs = GetTSRListForObject(object, manip)

  #turn TSR into poses
  num_poses_desired = 30
  max_num_poses_sample = 500

  target_poses = []
  target_iks = []
  num_sampled = 0
  while len(target_poses) < num_poses_desired and num_sampled < max_num_poses_sample:
    print 'name: ' + object.GetName() + ' currently has ' + str(len(target_poses)) + ' goal poses'
    if not 'bowl' in object.GetName():
      num_sample_this = int(math.ceil(num_poses_desired/len(target_tsrs)))
      num_sampled += num_sample_this
      target_poses_idenframe = SampleTSRList(target_tsrs, num_sample_this)
      target_poses_tocheck = [np.dot(object.GetTransform(), pose) for pose in target_poses_idenframe]
    else:
      num_sample_this = num_poses_desired
      num_sampled += num_sample_this
      target_poses_tocheck = get_bowl_poses(object, num_samples_pose=num_sample_this, ee_offset=0.15)
    for pose in target_poses_tocheck:
      #check if solution exists
#      ik_sols = manip.FindIKSolutions(pose, openravepy.IkFilterOptions.CheckEnvCollisions)
#      if len(ik_sols) > 0:

      
      #sample some random joint vals
      
#      lower, upper = robot.GetDOFLimits()
#      dof_vals_before = robot.GetActiveDOFValues()
#      dof_vals = [ np.random.uniform(lower[i], upper[i]) for i in range(6)]
#      robot.SetActiveDOFValues(dof_vals)
#      pose = manip.GetEndEffectorTransform()
#      robot.SetActiveDOFValues(dof_vals_before)

      ik_sol = manip.FindIKSolution(pose, openravepy.IkFilterOptions.CheckEnvCollisions)
      if ik_sol is not None:
        if ADD_MORE_IK_SOLS:
          #get bigger list of ik solutions
          ik_sols = manip.FindIKSolutions(pose, openravepy.IkFilterOptions.CheckEnvCollisions)
          if ik_sols is None:
            ik_sols = list()
          else:
            ik_sols = list(ik_sols)
          #add the solution we found before
          ik_sols.append(ik_sol)
        else:
          #if we don't want to add more, just use the one we found
          ik_sols = [ik_sol]
        #check env col
        target_poses.append(pose)
        target_iks.append(ik_sols)
#        with robot:
#          manip.SetDOFValues(ik_sol)
#          if not env.CheckCollision(robot):
#            target_poses.append(pose)
        if len(target_poses) >= num_poses_desired:
          break

  return Goal(pose, target_poses, target_iks)
