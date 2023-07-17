
from mcts_node import MCTSNode
from random import choice, randrange, shuffle
from math import sqrt, log

num_nodes = 50
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

def expand_leaf(node, board, state, identity_of_bot):
    """ Adds a new leaf to the tree by creating a new child node for the given node.

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:    The added child node.

    """
    if len(node.untried_actions) == 0:
        return node, state

    next_move = node.untried_actions.pop(randrange(0, len(node.untried_actions)))
    #print("chose:", next_move)
    #print("legal actions:", node.untried_actions, "chose:", next_move)
    #node.untried_actions.remove(next_move)
    new_node = MCTSNode(parent=node, parent_action=next_move, action_list=node.untried_actions)
    node.child_nodes[next_move] = new_node
    state = board.next_state(state, next_move)
    return new_node, state


def pick_next_action(board, state, legal_actions, identity_of_bot):
    return get_action_center_heuristic(board, state, legal_actions)
    #return get_action_simple_heuristic(board, state, legal_actions, identity_of_bot)
    #return get_action_length_heuristic(board, state, legal_actions)
    #return get_action_line_heuristic(board, state, legal_actions)
    #return choice(legal_actions)
    #return get_action_basic_heuristic(board, state, legal_actions)
    #return get_action_box_focus(board, state, legal_actions, identity_of_bot)

    '''
    return get_action_smart_heuristic_from_score(board, state, legal_actions, identity_of_bot)

    return get_action_simple_heuristic(board, state, legal_actions, identity_of_bot)

    return choice(legal_actions)

    if board.current_player(state) != identity_of_bot:
        return choice(legal_actions)
    opponent = 3 - identity_of_bot #(-1 if v == opponent else 0)
    current_owned_boxes = sum(1 if v == identity_of_bot else 0 for v in board.owned_boxes(state).values())
    for action in legal_actions:
        new_state = board.next_state(state, action)
        new_owned_boxes = sum(1 if v == identity_of_bot else 0 for v in board.owned_boxes(new_state).values())
        if new_owned_boxes >= current_owned_boxes:
            return action
    return choice(legal_actions)
    '''

    #return smart_heuristic(board, state, legal_actions, identity_of_bot)

def get_action_center_heuristic(board, state, legal_actions):
    for row in range(3):
        for col in range(3):
            subboard = (row, col)
            center_action = (subboard, (1, 1))  # Calculate the center position for the subboard
            if center_action in legal_actions:
                return center_action
    return choice(legal_actions)
def get_action_simple_heuristic(board, state, legal_actions, identity_of_bot):
    def outcome(owned_boxes, game_points):
        opponent = 3 - identity_of_bot
        if game_points is not None:
            my_score = game_points[identity_of_bot]*5
            their_score = game_points[opponent]*5
        else:
            my_score = len([v for v in owned_boxes.values() if v == identity_of_bot])
            their_score = len([v for v in owned_boxes.values() if v == opponent])
        return my_score - their_score

    scores = {}
    for action in legal_actions:
        new_state = board.next_state(state, action)
        scores[action] = outcome(board.owned_boxes(new_state), board.points_values(new_state))

    return max(scores, key=scores.get)

def length_heuristic(board, state):
    """ Heuristic evaluation function for the game state.

    Args:
        board: The game setup.
        state: The state of the game.

    Returns:
        score: The heuristic score for the given state.
    """
    return len(board.legal_actions(state))

def get_action_length_heuristic(board, state, legal_actions):
    scores = {}
    for action in legal_actions:
        new_state = board.next_state(state, action)
        scores[action] = length_heuristic(board, new_state)
    return max(scores, key=scores.get)


def get_action_line_heuristic(board, state, legal_actions):
    scores = {}
    shuffle(legal_actions)
    for action in legal_actions:
        new_state = board.next_state(state, action)
        board_pieces = board.unpack_state(new_state)["pieces"]
        board_pos = action[0], action[1]

        grid = get_grid(board_pieces, board_pos)
        if len(grid) == 0:
            return choice(legal_actions)

        for x in range(0, 3):
            lineCount = 0
            for y in range(0, 3):
                pos = board_pos[0], board_pos[1], x, y
                if pos in grid:
                    lineCount += 1
            add_to_dict_val(scores, action, pow(lineCount, 1.5) / 2)
            '''
        for y in range(0, 3):
            lineCount = 0
            for x in range(0, 3):
                pos = board_pos[0], board_pos[1], x, y
                if pos in grid:
                    lineCount += 1
            add_to_dict_val(scores, action, pow(lineCount, 1.5) / 2)
            '''
    #print(scores)
    best = max(scores, key=scores.get)
    #print("best:", best)
    return best

def get_grid(board_pieces, board_pos):
    grid = {}
    for piece in board_pieces:
        pos = piece['outer-row'], piece['outer-column'], piece['inner-row'], piece['inner-column']
        if pos[0] == board_pos[0] and pos[1] == board_pos[1]:
            grid[pos] = piece['player']
    return grid

def heuristic(board, state):
    """ Heuristic evaluation function for the game state.

    Args:
        board: The game setup.
        state: The state of the game.

    Returns:
        score: The heuristic score for the given state.
    """
    # Get the current player and the opponent.
    player = board.current_player(state)
    opponent = 3 - player

    # Check the ownership of each small board.
    ownership = board.owned_boxes(state)

    # Initialize the score to 0.
    score = 0

    # Add 1 point for each small board owned by the current player.
    # Subtract 1 point for each small board owned by the opponent.
    for owner in ownership.values():
        if owner == player:
            score += 1
        elif owner == opponent:
            score -= 1

    # Add 0.1 points for each potential winning line in a small board.
    for i in range(9):
        for win in board.wins:
            small_board = (state[2*i], state[2*i+1])
            if win & small_board[player-1] and not win & small_board[opponent-1]:
                score += 0.1

    return score

def get_action_basic_heuristic(board, state, legal_actions):
    action_scores = {}
    for action in legal_actions:
        next_state = board.next_state(state, action)
        score = heuristic(board, next_state)
        action_scores[action] = score

    # Choose the action with the highest score
    return max(action_scores, key=action_scores.get)

def get_action_box_focus(board, state, legal_actions, identity_of_bot):
    #if board.current_player(state) != identity_of_bot:

    unpacked_board = board.unpack_state(state)
    pieces = unpacked_board["pieces"]
    boards = unpacked_board["boards"]

    piece_occurrences = {}
    for piece in pieces:
        box = piece['outer-row'], piece['outer-column']
        is_mine = piece['player'] == identity_of_bot
        if box not in piece_occurrences:
            piece_occurrences[box] = int(is_mine), int(not is_mine)
        else:
            current_val = piece_occurrences[box]
            piece_occurrences[box] = current_val[0] + int(is_mine), current_val[1] + int(not is_mine)
    print(board.display(state, None))
    print("boards:", boards)
    desired_target_box, occurrences = None, 10
    for item in piece_occurrences.items():
        print(item)
        opponent_occurrences = item[1][1]
        if opponent_occurrences < occurrences and boards:
            desired_target_box = item[0]
            occurrences = opponent_occurrences


    print(desired_target_box)

    box_action_count = {}
    for action in legal_actions:
        box = action[0], action[1]
        if box not in box_action_count:
            box_action_count[box] = 1
        else:
            box_action_count[box] += 1
    best_box = min(box_action_count, key=box_action_count.get)
    actions_in_best_box = list(v for v in legal_actions if v[0] == best_box[0] and v[1] == best_box[1])

    #for action in actions_in_best_box:
    #    if action[2] == desired_target_box[]

    return choice(actions_in_best_box)
    return choice(legal_actions)

def get_action_smart_heuristic(board, state, legal_actions, identity_of_bot):
    if board.current_player(state) != identity_of_bot:
        return choice(legal_actions)

    box_action_occurrences = {}
    for action in legal_actions:
        box = action[0], action[1]
        if box not in box_action_occurrences:
            box_action_occurrences[box] = 1
        else:
            box_action_occurrences[box] += 1

    most_populated_box = min(box_action_occurrences, key=box_action_occurrences.get)
    board_pos = most_populated_box[0], most_populated_box[1]#legal_actions[0][0], legal_actions[0][1]
    board_pieces = board.unpack_state(state)["pieces"]

    flipped_legal_actions = [(x, y) for (Y, X, y, x) in legal_actions if Y == board_pos[0] and X == board_pos[1]]
    grid = {}

    for piece in board_pieces:
        # print("piece:", piece)
        if piece['outer-row'] == board_pos[0] and piece['outer-column'] == board_pos[1]:
            pos = (piece['inner-column'], piece['inner-row'])
            grid[pos] = piece['player']
    # print(grid)

    # print("adding", pos)
    # board_pieces = filter(lambda p: p['outer-row'] == legal_actions[0][0] and p['outer-column'] == legal_actions[0][1], board_pieces)

    best_local_pos = None
    neighbor_counts = {}

    positions = list(grid.keys())
    shuffle(positions)

    for pos in positions:
        #print("pos:", pos, grid[pos])
        is_me = grid[pos] == identity_of_bot
        for x in range(-1, 2):
            for y in range(-1, 2):
                neighbor1, neighbor2 = get_wrapped_neighbor_positions(pos, x, y)
                #print(pos, neighbor1, neighbor2, "is me?", is_me, "best:", best_local_pos)
                hasNeighbor1 = neighbor1 in grid
                hasNeighbor2 = neighbor2 in grid

                if not hasNeighbor1:
                    added_value = int(is_me), int(not is_me)
                    if neighbor1 not in neighbor_counts:
                        neighbor_counts[neighbor1] = added_value
                    else:
                        curr = neighbor_counts[neighbor1]
                        neighbor_counts[neighbor1] = curr[0] + added_value[0], curr[1] + added_value[1]
                #if neighbor1 in neighbor_counts:
                #    print("{}: {}".format(neighbor1, neighbor_counts[neighbor1]))
                #else:
                #    print("{}: none".format(neighbor1))

                if x == 0 and y == 0 or (abs(x) == 1 and abs(y) == 1 and (pos[0] + x != 1 or pos[1] + y != 1)):  # and (pos[0] % 2 != 0 and pos[1] % 2 != 0):
                    continue

                if is_me:
                    if hasNeighbor1 and grid[neighbor1] == identity_of_bot and neighbor2 in flipped_legal_actions:
                        return board_pos[0], board_pos[1], neighbor2[1], neighbor2[0]
                    if hasNeighbor2 and grid[neighbor2] == identity_of_bot and neighbor1 in flipped_legal_actions:
                        return board_pos[0], board_pos[1], neighbor1[1], neighbor1[0]
                    if best_local_pos is None:
                        if neighbor1 in flipped_legal_actions and not hasNeighbor2:
                            best_local_pos = neighbor1
                        elif neighbor2 in flipped_legal_actions and not hasNeighbor1:
                            best_local_pos = neighbor2
                else:
                    if hasNeighbor1 and grid[neighbor1] != identity_of_bot and neighbor2 in flipped_legal_actions:
                        return board_pos[0], board_pos[1], neighbor2[1], neighbor2[0]
                    if hasNeighbor2 and grid[neighbor2] != identity_of_bot and neighbor1 in flipped_legal_actions:
                        return board_pos[0], board_pos[1], neighbor1[1], neighbor1[0]

    #print(neighbor_counts)
    if best_local_pos is not None:
        #print("picking best choice")
        return board_pos[0], board_pos[1], best_local_pos[1], best_local_pos[0]

    #center_pos = board_pos[0], board_pos[1], 1, 1
    #if center_pos in legal_actions:
    #    return center_pos
    #print("random choice")
    return choice(legal_actions)

def get_action_smart_heuristic_from_score(board, state, legal_actions, identity_of_bot):
    if board.current_player(state) != identity_of_bot:
        return choice(legal_actions)

    box_action_occurrences = {}
    for action in legal_actions:
        box = action[0], action[1]
        if box not in box_action_occurrences:
            box_action_occurrences[box] = 1
        else:
            box_action_occurrences[box] += 1

    most_populated_box = min(box_action_occurrences, key=box_action_occurrences.get)
    board_pos = most_populated_box[0], most_populated_box[1]#legal_actions[0][0], legal_actions[0][1]
    board_pieces = board.unpack_state(state)["pieces"]

    flipped_legal_actions = [(x, y) for (Y, X, y, x) in legal_actions if Y == board_pos[0] and X == board_pos[1]]
    grid = {}

    for piece in board_pieces:
        # print("piece:", piece)
        if piece['outer-row'] == board_pos[0] and piece['outer-column'] == board_pos[1]:
            pos = (piece['inner-column'], piece['inner-row'])
            grid[pos] = piece['player']
    # print(grid)

    # print("adding", pos)
    # board_pieces = filter(lambda p: p['outer-row'] == legal_actions[0][0] and p['outer-column'] == legal_actions[0][1], board_pieces)

    best_local_pos = None
    neighbor_counts = {}

    positions = list(grid.keys())
    shuffle(positions)

    scores = {}

    for pos in positions:
        #print("pos:", pos, grid[pos])
        is_me = grid[pos] == identity_of_bot
        for x in range(-1, 2):
            for y in range(-1, 2):
                neighbor1, neighbor2, n1wraps, n2wraps = get_wrapped_neighbor_positions(pos, x, y)
                print("{}: {}({}), {}({}), is_me: {}".format(pos, neighbor1, n1wraps, neighbor2, n2wraps, is_me))
                hasNeighbor1 = neighbor1 in grid
                hasNeighbor2 = neighbor2 in grid

                if not hasNeighbor1 and not n1wraps:
                    add_to_dict_val(scores, neighbor1, 0.125)
                #if neighbor1 in neighbor_counts:
                #    print("{}: {}".format(neighbor1, neighbor_counts[neighbor1]))
                #else:
                #    print("{}: none".format(neighbor1))

                if x == 0 and y == 0 or (abs(x) == 1 and abs(y) == 1 and (pos[0] + x != 1 or pos[1] + y != 1)):  # and (pos[0] % 2 != 0 and pos[1] % 2 != 0):
                    continue

                if is_me:
                    if hasNeighbor1 and grid[neighbor1] == identity_of_bot and neighbor2 in flipped_legal_actions:
                        add_to_dict_val(scores, (board_pos[0], board_pos[1], neighbor2[1], neighbor2[0]), 1)
                    if hasNeighbor2 and grid[neighbor2] == identity_of_bot and neighbor1 in flipped_legal_actions:
                        add_to_dict_val(scores, (board_pos[0], board_pos[1], neighbor1[1], neighbor1[0]), 1)
                    #if best_local_pos is None:
                    #    if neighbor1 in flipped_legal_actions and not hasNeighbor2:
                    #        best_local_pos = neighbor1
                    #    elif neighbor2 in flipped_legal_actions and not hasNeighbor1:
                    #        best_local_pos = neighbor2
                else:
                    if hasNeighbor1 and grid[neighbor1] != identity_of_bot and neighbor2 in flipped_legal_actions:
                        add_to_dict_val(scores, (board_pos[0], board_pos[1], neighbor2[1], neighbor2[0]), 1.125)
                    if hasNeighbor2 and grid[neighbor2] != identity_of_bot and neighbor1 in flipped_legal_actions:
                        add_to_dict_val(scores, (board_pos[0], board_pos[1], neighbor1[1], neighbor1[0]), 1.125)

    if len(scores) > 0:
        best = max(scores, key=scores.get)
        return board_pos[0], board_pos[1], best[1], best[0]

    return choice(legal_actions)

def add_to_dict_val(dictionary, key, val):
    if key not in dictionary:
        dictionary[key] = val
    else:
        curr = dictionary[key]
        dictionary[key] = curr + val

def get_wrapped_neighbor_positions(pos, x_offset, y_offset):
    n1_wraps = False
    raw_x1 = pos[0] + x_offset
    raw_y1 = pos[1] + y_offset
    if raw_x1 > 2 or raw_y1 > 2:
        n1_wraps = True
    neighbor_x1 = raw_x1 % 3
    neighbor_y1 = raw_y1 % 3
    if raw_x1 == -1:
        neighbor_x1 = 2
        n1_wraps = True
    if raw_y1 == -1:
        neighbor_y1 = 2
        n1_wraps = True
    n2_wraps = False
    raw_x2 = pos[0] - x_offset
    raw_y2 = pos[1] - y_offset
    if raw_x2 > 2 or raw_y2 > 2:
        n2_wraps = True
    neighbor_x2 = raw_x2 % 3
    neighbor_y2 = raw_y2 % 3
    if raw_x2 == -1:
        neighbor_x2 = 2
        n2_wraps = True
    if raw_y2 == -1:
        neighbor_y2 = 2
        n2_wraps = True
    return (neighbor_x1, neighbor_y1), (neighbor_x2, neighbor_y2), n1_wraps, n2_wraps

    #enemy_pieces = filter(lambda p: p['player'] != identity_of_bot, board_pieces)
    #for enemy_piece in enemy_pieces:


def sort_pieces_by_player(e):
    return e['player']
def rollout(board, state, identity_of_bot):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.

    """

    while board.is_ended(state) is False:
        legal_actions = board.legal_actions(state)
        best_action = pick_next_action(board, state, legal_actions, identity_of_bot)
        #print("best action:", best_action)
        state = board.next_state(state, best_action)
    return state

def alt_rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder using a heuristic-based strategy.

    Args:
        board: The game setup.
        state: The state of the game.

    Returns:
        The final state of the game.
    """
    while not board.is_ended(state):
        legal_actions = board.legal_actions(state)

        # Iterate through each subboard and check the center position
        for row in range(3):
            for col in range(3):
                subboard = (row, col)
                center_action = (subboard, (1, 1))  # Calculate the center position for the subboard
                if center_action in legal_actions:
                    state = board.next_state(state, center_action)
                    break  # Exit the loop if the center position is played

            else:
                continue
            break

        else:
            # If no center position is available in any subboard, choose a random action
            action = choice(legal_actions)
            state = board.next_state(state, action)

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


def get_rollout_action(board, state, rollouts, depth):
    moves = board.legal_actions(state)

    best_move = moves[0]
    best_expectation = float('-inf')

    me = board.current_player(state)

    # Define a helper function to calculate the difference between the bot's score and the opponent's.
    def outcome(owned_boxes, game_points):
        if game_points is not None:
            # Try to normalize it up?  Not so sure about this code anyhow.
            red_score = game_points[1] * 9
            blue_score = game_points[2] * 9
        else:
            red_score = len([v for v in owned_boxes.values() if v == 1])
            blue_score = len([v for v in owned_boxes.values() if v == 2])
        return red_score - blue_score if me == 1 else blue_score - red_score

    for move in moves:
        total_score = 0.0

        # Sample a set number of games where the target move is immediately applied.
        for r in range(rollouts):
            rollout_state = board.next_state(state, move)

            # Only play to the specified depth.
            for i in range(depth):
                if board.is_ended(rollout_state):
                    break
                rollout_move = choice(board.legal_actions(rollout_state))
                rollout_state = board.next_state(rollout_state, rollout_move)

            total_score += outcome(board.owned_boxes(rollout_state),
                                   board.points_values(rollout_state))

        expectation = float(total_score) / rollouts

        # If the current move has a better average score, replace best_move and best_expectation
        if expectation > best_expectation:
            best_expectation = expectation
            best_move = move

    # print("Rollout bot picking %s with expected score %f" % (str(best_move), best_expectation))
    return best_move

def think(board, state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """

    identity_of_bot = board.current_player(state)
    #return pick_next_action(board, state, board.legal_actions(state), identity_of_bot)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))

    for step in range(num_nodes):
        # Selection
        #print("selecting...")
        node, sampled_game = traverse_nodes(root_node, board, state, identity_of_bot)

        # Expansion
        #print("expanding...")
        #if node.visits > 0:
        node, sampled_game = expand_leaf(node, board, sampled_game, identity_of_bot)

        # Evaluation/Simulation/Rollout
        #print("rolling out...")
        sampled_game = rollout(board, sampled_game, identity_of_bot)

        # Backpropagation
        #print("back propagating...")
        won = board.points_values(sampled_game)[identity_of_bot]

        backpropagate(node, won)
    #for v in root_node.child_nodes:
    #    print("{}: {}".format(v, root_node.child_nodes[v]))
    best_node = max(root_node.child_nodes.values(), key=lambda x: win_rate(x))
    action = best_node.parent_action
    #print("\nbest: {}\n".format(best_node))

    #print(win_rate(best_node))
    #print("board:")
    #print(board.display(state, action))

    return action
