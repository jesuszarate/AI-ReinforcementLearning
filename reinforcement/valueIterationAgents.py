# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        while iterations > 0:
          updates = {}
          for state in mdp.getStates():
            possibleActs = mdp.getPossibleActions(state)
            if len(possibleActs) > 0:
              maxActionValue = None
              for action in possibleActs:
                # v_i+1(state) = max_action (sum_nextState (T_action(state, nextState) * (R_action(state, nextState) + discout * V_i(nextState) )) )
                qValue = self.computeQValueFromValues(state, action)
                if maxActionValue == None or qValue > maxActionValue:
                  maxActionValue = qValue
              updates.update({state: maxActionValue})
          self.values.update(updates)
          iterations -= 1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qValue = 0
        for nextState, tranProb in self.mdp.getTransitionStatesAndProbs(state, action):
          qValue += tranProb * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))

        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
          return None

        # loops through all possible actions from provided state and finds
        # the nextState that has the maximum value. the action that results
        # in that nextState is returned
        bestVal = None
        bestAct = None

        acts = self.mdp.getPossibleActions(state)
        for action in acts:
          qValue = self.computeQValueFromValues(state, action)
          if bestVal == None or qValue > bestVal:
            bestVal = qValue
            bestAct = action

        if not bestAct and 'exit' in acts:
          bestAct = 'exit'

        return bestAct

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
