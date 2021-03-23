import pydot

from decisiontree.handcrafted_trees import scenario_trees
from core.base import get_img_dir


def build_pydot_tree(graph, root, idx='R'):
    node = pydot.Node(idx, label=str(root))
    graph.add_node(node)
    if root.decision is not None:
        true_child = build_pydot_tree(graph, root.decision.true_child, idx + 'T')
        false_child = build_pydot_tree(graph, root.decision.false_child, idx + 'F')
        true_weight = root.decision.true_child.value / root.value
        false_weight = root.decision.false_child.value / root.value
        graph.add_edge(pydot.Edge(node, true_child, label='T: {:.2f}'.format(true_weight)))
        graph.add_edge(pydot.Edge(node, false_child, label='F: {:.2f}'.format(false_weight)))
    return node


scenario_name = 'heckstrasse'

for goal_idx, goal_types in scenario_trees[scenario_name].items():
    for goal_type, root in goal_types.items():
        graph = pydot.Dot(graph_type='digraph')
        build_pydot_tree(graph, root)
        graph.write_png(get_img_dir() + 'handcrafted_tree_{}_G{}_{}.png'.format(
            scenario_name, goal_idx, goal_type))
