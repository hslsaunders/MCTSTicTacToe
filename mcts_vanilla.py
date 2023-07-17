from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

num_nodes = 1000
explore_factor = 2.0


def traverse_nodes(node, board, state, identity):
    """ Traverses the tree until the end criterion is met.

    Args:
        node: A tree node from which the search is traversing.
        board: The game setup.
        state: The state of the game.
        identity: The bot's identity, either 'red' or 'blue'.

    Returns: A node from which the next stage of the search can proceed.
    """
    high_score = float('-inf')
    best_node = None

    for child_node in node.child_nodes.values():
        if child_node.visits == 0:
            return child_node

        score = (child_node.wins / child_node.visits) + explore_factor * sqrt(log(node.visits) / child_node.visits)

        if score > high_score:
            high_score = score
            best_node = child_node

    return best_node


def expand_leaf(node, board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node.

    Args:
        node: The node for which a child will be added.
        board: The game setup.
        state: The state of the game.

    Returns: The added child node.
    """
    untried_actions = node.untried_actions
    action = choice(untried_actions)
    untried_actions.remove(action)
    child_state = board.next_state(state, action)
    child_node = MCTSNode(parent=node, parent_action=action, action_list=untried_actions)
    node.child_nodes[action] = child_node
    return child_node


def rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board: The game setup.
        state: The state of the game.
    """
    while not board.is_ended(state):
        legal_actions = board.legal_actions(state)
        action = choice(legal_actions)
        state = board.next_state(state, action)
    return state

def backpropagate(node, won):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node: A leaf node.
        won: An indicator of whether the bot won or lost the game.
    """
    while node is not None:
        node.visits += 1
        if won:
            node.wins += 1
        node = node.parent


def think(board, state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board: The game setup.
        state: The state of the game.

    Returns: The action to be taken.
    """
    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))

    for step in range(num_nodes):
        sampled_game = state

        node = root_node

        while node.untried_actions == [] and node.child_nodes != {}:
            node = traverse_nodes(node, board, sampled_game, identity_of_bot)
            action = node.parent_action
            sampled_game = board.next_state(sampled_game, action)

        if node.untried_actions != []:
            node = expand_leaf(node, board, sampled_game)
            action = node.parent_action
            sampled_game = board.next_state(sampled_game, action)

        final_state = rollout(board, sampled_game)
        won = board.is_ended(final_state) and board.win_values(final_state)[identity_of_bot] == 1.0
        backpropagate(node, won)

    best_action = None
    best_score = float('-inf')

    for child_action, child_node in root_node.child_nodes.items():
        child_score = child_node.wins / child_node.visits

        if child_score > best_score:
            best_score = child_score
            best_action = child_action

    return best_action
