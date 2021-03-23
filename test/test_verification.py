import pandas as pd
import numpy as np
from z3 import *

from decisiontree.decision_tree import Node, BinaryDecision, ThresholdDecision
from decisiontree.dt_goal_recogniser import DecisionTreeGoalRecogniser
from core.scenario import Scenario, ScenarioConfig, Frame, AgentState
from evaluation.verification import add_goal_tree_model
from test.lanelet_test_helpers import get_test_map


class MockScenario(Scenario):

    def __init__(self, config):
        self.config = config
        self.lanelet_map = get_test_map()


def get_goal_tree_model():
    goal_priors = pd.DataFrame({'true_goal': [0, 1],
                                'true_goal_type': ['straight-on', 'turn-left'],
                                'prior': [0.75, 0.25]})
    config = ScenarioConfig({'goals': [[3.5, 0.5], [3.0, 2.5]]})
    scenario = MockScenario(config)

    decision_trees = {0: {'straight-on': Node(0.5, BinaryDecision('in_correct_lane',
                                                                  Node(0.8, ThresholdDecision(0.1, 'angle_in_lane',
                                                                                              Node(0.2),
                                                                                              Node(0.8)
                                                                                              )),
                                                                  Node(0.2)
                                                                  )
                                              )
                          },
                      1: {'turn-left': Node(0.5, BinaryDecision('in_correct_lane',
                                                                Node(0.8),
                                                                Node(0.2)
                                                                )
                                            )
                          }
                      }

    model = DecisionTreeGoalRecogniser(goal_priors, scenario, decision_trees)
    return model


def test_goal_tree_model():
    model = get_goal_tree_model()
    frame = Frame(0)
    state = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame.add_agent_state(0, state)
    frames = [frame]
    probs = model.goal_probabilities(frames, 0)
    assert np.allclose(probs, [0.92307692, 0.07692308])


def test_goal_tree_model_2():
    model = get_goal_tree_model()
    frame0 = Frame(0)
    state0 = AgentState(0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame0.add_agent_state(0, state0)
    frame1 = Frame(1)
    state1 = AgentState(1, 0.5, 0.5, 0, 0, np.pi/8, 0, 0, 0, 0, 0, 0)
    frame1.add_agent_state(0, state1)
    frames = [frame0, frame1]
    probs = model.goal_probabilities(frames, 0)
    assert np.allclose(probs, [0.75, 0.25])


def approx_equal(a, b, tol=1e-4):
    return And(a - b < tol, b - a < tol)


def test_z3_model_g0():
    s = Solver()
    model = get_goal_tree_model()
    reachable_goals = [(0, 'straight-on'), (1, 'turn-left')]
    features, probs, likelihoods = add_goal_tree_model(reachable_goals, s, model)

    verify_expr = Implies(And(features[0]['in_correct_lane'],
                              Not(features[1]['in_correct_lane']),
                              features[0]['angle_in_lane'] == 0
                              ),
                          And(approx_equal(probs[0], 12/13),
                              approx_equal(probs[1], 1/13))
                          )
    s.add(Not(verify_expr))
    assert str(s.check()) == 'unsat'


def test_z3_model_g1():
    s = Solver()
    model = get_goal_tree_model()
    reachable_goals = [(0, 'straight-on'), (1, 'turn-left')]
    features, probs, likelihoods = add_goal_tree_model(reachable_goals, s, model)

    verify_expr = Implies(And(features[0]['in_correct_lane'],
                              Not(features[1]['in_correct_lane']),
                              features[0]['angle_in_lane'] == np.pi/8
                              ),
                          And(approx_equal(probs[0], 0.75),
                              approx_equal(probs[1], 0.25))
                          )
    s.add(Not(verify_expr))
    assert str(s.check()) == 'unsat'

