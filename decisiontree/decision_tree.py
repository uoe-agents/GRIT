import pickle

import pydot
from sklearn.tree import _tree

from core.feature_extraction import FeatureExtractor


class Node:
    def __init__(self, value, decision=None):
        self.value = value
        self.decision = decision
        self.counts = [None, None]
        self.reached = False

    def traverse(self, features):
        self.reached = True
        current_node = self
        while current_node.decision is not None:
            current_node = current_node.decision.select_child(features)
            current_node.reached = True
        return current_node.value

    def reset_reached(self):
        self.reached = False
        if self.decision is not None:
            self.decision.true_child.reset_reached()
            self.decision.false_child.reset_reached()

    def __str__(self):
        text = ''
        #text += '{0:.3f} {1}\n'.format(self.value, self.counts)
        text += '{0:.3f}\n'.format(self.value)
        if self.decision is not None:
            text += str(self.decision)
        return text

    def get_depth(self):
        depth = 0
        if self.decision is not None:
            depth += 1 + max(self.decision.true_child.get_depth(), self.decision.false_child.get_depth())
        return depth

    @classmethod
    def from_sklearn(cls, input_tree, feature_types):
        # based on:
        # https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

        tree_ = input_tree.tree_
        feature_names = [*feature_types]
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node):
            value = tree_.value[node][0][1] / tree_.value[node].sum()
            out_node = Node(value)
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                true_child = recurse(tree_.children_right[node])
                false_child = recurse(tree_.children_left[node])
                if feature_types[name] == 'scalar':
                    out_node.decision = ThresholdDecision(threshold, name, true_child, false_child)
                elif feature_types[name] == 'binary':
                    out_node.decision = BinaryDecision(name, true_child, false_child)
                else:
                    raise ValueError('invalid feature type')
            return out_node

        return recurse(0)

    def set_values(self, training_samples, goal, alpha=0):

        goal_training_samples = training_samples.loc[training_samples.possible_goal == goal]
        N = goal_training_samples.shape[0]
        Ng = (goal_training_samples.true_goal == goal).sum()
        goal_normaliser = (N + 2 * alpha) / 2 / (Ng + alpha)
        non_goal_normaliser = (N + 2 * alpha) / 2 / (N - Ng + alpha)
        feature_names = [*FeatureExtractor.feature_names]

        def recurse(node, node_samples):
            Nng = node_samples.loc[node_samples.true_goal == goal].shape[0]
            Nn = node_samples.shape[0]
            Nng_norm = (Nng + alpha) * goal_normaliser
            Nn_norm = Nng_norm + (Nn - Nng + alpha) * non_goal_normaliser
            value = Nng_norm / Nn_norm
            node.value = value
            node.counts = [Nng, Nn - Nng]
            features = node_samples.loc[:, feature_names]
            if node.decision is not None:
                rule_true = node.decision.rule(features)
                true_child_samples = node_samples.loc[rule_true]
                false_child_samples = node_samples.loc[~rule_true]
                recurse(node.decision.true_child, true_child_samples)
                recurse(node.decision.false_child, false_child_samples)

        recurse(self, goal_training_samples)

    def pydot_tree(self):
        graph = pydot.Dot(graph_type='digraph')

        def recurse(graph, root, idx='R'):
            if root.reached:
                node = pydot.Node(idx, label=str(root),  style='filled', color="lightblue")
            else:
                node = pydot.Node(idx, label=str(root))
            graph.add_node(node)
            if root.decision is not None:
                true_child = recurse(graph, root.decision.true_child, idx + 'T')
                false_child = recurse(graph, root.decision.false_child, idx + 'F')
                true_weight = root.decision.true_child.value / root.value
                false_weight = root.decision.false_child.value / root.value
                graph.add_edge(pydot.Edge(node, true_child, label='T: {:.2f}'.format(true_weight)))
                graph.add_edge(pydot.Edge(node, false_child, label='F: {:.2f}'.format(false_weight)))
            return node

        recurse(graph, self)
        return graph

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class Decision:

    def __init__(self, feature_name, true_child, false_child):
        self.feature_name = feature_name
        self.true_child = true_child
        self.false_child = false_child

    def rule(self, features):
        raise NotImplementedError

    def select_child(self, features):
        if self.rule(features):
            return self.true_child
        else:
            return self.false_child


class BinaryDecision(Decision):

    def rule(self, features):
        return features[self.feature_name]

    def __str__(self):
        return self.feature_name + '\n'


class ThresholdDecision(Decision):

    def __init__(self, threshold, *args):
        super().__init__(*args)
        self.threshold = threshold

    def rule(self, features):
        return features[self.feature_name] > self.threshold

    def __str__(self):
        return '{} > {:.2f}\n'.format(self.feature_name, self.threshold)

