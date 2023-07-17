
from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

num_nodes = 1000
explore_factor = 2.0

def traverse_nodes(node, board, state, identity):
    """ Traverses the tree until the end criterion are met.

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 'red' or 'blue'.

    Returns:        A node from which the next stage of the search can proceed.

    """

    best_child = node
    highest_score = float('-inf')

    while len(node.child_nodes) > 0 and len(node.untried_actions) == 0:
        for child_action in node.child_nodes:
            child_node = node.child_nodes[child_action]
            score = calc_uct(node, child_node, board.current_player(state) == identity)
            if score > highest_score:
                highest_score = score
                best_child = child_node

        #best_child = max(node.child_nodes.values(), key=lambda x: calc_uct(node, x, board.current_player(state) == identity))
        if node is best_child:
            break
        node = best_child
        if node.parent_action is not None:
            state = board.next_state(state, node.parent_action)

    return best_child, state

def expand_leaf(node, board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node.

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:    The added child node.

    """
    if len(node.untried_actions) == 0:
        return node, state

    next_move = node.untried_actions.pop(0)
    new_node = MCTSNode(parent=node, parent_action=next_move, action_list=node.untried_actions)
    node.child_nodes[next_move] = new_node
    state = board.next_state(state, next_move)
    return new_node, state


def rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.

    """

    while board.is_ended(state) is False:
        state = board.next_state(state, choice(board.legal_actions(state)))
    return state
def backpropagate(node, won):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    node.wins += won
    node.visits += 1
    if node.parent is not None:
        backpropagate(node.parent, won)

def calc_uct(parent, child, is_player_turn):
    return ((win_rate(child) if is_player_turn else (1 - win_rate(child))) + explore_factor * sqrt(log(parent.visits) / child.visits)) if child.visits != 0 else 0

def win_rate(node):
    return node.wins / node.visits

def think(board, state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """
    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))

    for step in range(num_nodes):
        # Selection
        #print("selecting...")
        node, sampled_game = traverse_nodes(root_node, board, state, identity_of_bot)

        # Expansion
        #print("expanding...")
        if node.visits > 0:
            node, sampled_game = expand_leaf(node, board, sampled_game)

        # Evaluation/Simulation/Rollout
        #print("rolling out...")
        sampled_game = rollout(board, sampled_game)

        # Backpropagation
        #print("back propagating...")
        won = board.points_values(sampled_game)[identity_of_bot]

        backpropagate(node, won)

    return max(root_node.child_nodes.values(), key=lambda x: win_rate(x)).parent_action
