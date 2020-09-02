#Generic assistance policy for one goal
import numpy as np
import IPython
import AssistancePolicyOneGoal as GoalPolicy

class AssistancePolicy:

  def __init__(self, goals):
    self.goals = goals
    self.goal_assist_policies = []
    for goal in goals:
      self.goal_assist_policies.append(GoalPolicy.AssistancePolicyOneGoal(goal))

  def update(self, robot_state, user_action):
    self.robot_state = robot_state
    #user action corresponds to the effect of direct teleoperation on the robot
    #self.user_action = self.user_input_mapper.input_to_action(user_input, robot_state)
    self.user_action = user_action

    for goal_policy in self.goal_assist_policies:
      goal_policy.update(robot_state, self.user_action)

  def get_values(self):
    values = np.ndarray(len(self.goal_assist_policies))
    qvalues = np.ndarray(len(self.goal_assist_policies))
    for ind,goal_policy in enumerate(self.goal_assist_policies):
      values[ind] = goal_policy.get_value()
      qvalues[ind] = goal_policy.get_qvalue()

    return values,qvalues

  def get_novalues(self):
    qnovalues = np.ndarray(len(self.goal_assist_policies))
    qvalues = np.ndarray(len(self.goal_assist_policies))
    for ind,goal_policy in enumerate(self.goal_assist_policies):
      qnovalues[ind] = goal_policy.get_qnovalue()
      qvalues[ind] = goal_policy.get_qvalue()

    return qnovalues,qvalues

  def get_robot_actions(self):
      return [ g.get_action() for g in self.goal_assist_policies ]

    
      
    

