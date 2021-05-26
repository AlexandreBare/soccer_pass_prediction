import time
import datetime
import sys
from contextlib import contextmanager
from math import sqrt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

N_PLAYERS = 22
MAX_X = 5250 # the max X coordinate without being out of the field
MAX_Y = 3400 # the max Y coordinate without being out of the field
MAX_DIST = sqrt((MAX_X * 2)**2 + (MAX_Y * 2)**2) # the distance of the diagonal of the field
CLOSE_DIST = MAX_Y # the distance between 2 players from which we consider them to be close to each other
VERY_CLOSE_DIST = MAX_DIST * 0.10
MAX_TIME = (45 + 10) * 60 * 1000 # 45 min of play + 10 min supp = Max time of a half period in a match

FEATURES = ["same_team", "region_player_j", "region_sender",
            "time_start", "distance", "manhattan_distance", "distance_time", "x_direction", "y_direction",
            "n_opponents_in_between", "n_opponents_in_between2", "n_close_opponents",
            "dist_to_bulk_opponents_score", "angle_score", "distance_opponent_pass",
            "distance_closest_opponent_sender", "distance_closest_opponent_player_j",
            "nb_teammate_in_front", "nb_opponents_in_front", "distance_second_closest_opponent_sender",
            "distance_second_closest_opponent_player_j"]

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds = end - start)))


def load_from_csv(path, delimiter = ','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter = delimiter)


def load_dataset():
    X_LS = load_from_csv('input_training_set.csv')
    y_LS = load_from_csv('output_training_set.csv')
    return X_LS, y_LS


def load_training_and_testing_data(random_state = 1):
    # Load training data
    X_LS = load_from_csv('input_training_set.csv')
    y_LS = load_from_csv('output_training_set.csv')

    # Train - Test Set split
    X_LS, X_TS, y_LS, y_TS = train_test_split(X_LS, y_LS, test_size = 0.3, random_state = random_state)
    return X_LS, y_LS, X_TS, y_TS


def same_team_(sender, player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)


def make_pair_of_players(X_, y_ = None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team"]
    pair_feature_col += [feature for feature in FEATURES if feature not in pair_feature_col]
    X_pairs = pd.DataFrame(data = np.zeros((n_ * 22, len(pair_feature_col))), columns = pair_feature_col)
    y_pairs = pd.DataFrame(data = np.zeros((n_ * 22, 1)), columns = ["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)

        p_i_ = X_.iloc[i]
        for player_j in players:
            X_pairs.iloc[idx] = [sender, p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j),
                                 compute_region(p_i_, player_j),
                                 compute_region(p_i_, sender),
                                 p_i_["time_start"],
                                 compute_distance_(p_i_, sender, player_j),
                                 compute_manhattan_distance_(p_i_, sender, player_j),
                                 compute_distance_time_(p_i_, sender, player_j),
                                 compute_x_direction_(p_i_, sender, player_j),
                                 compute_y_direction_(p_i_, sender, player_j),
                                 compute_n_opponents_in_between_(p_i_, sender, player_j),
                                 compute_n_opponents_in_between_2_(p_i_, sender, player_j),
                                 compute_n_close_opponents_(p_i_, player_j),
                                 compute_dist_to_bulk_opponents_score(p_i_, sender, player_j),
                                 compute_angle_score_(p_i_, sender, player_j),
                                 compute_distance_opponent_pass_(p_i_, sender, player_j),
                                 compute_distance_closest_opponent_(p_i_, sender),
                                 compute_distance_closest_opponent_(p_i_, player_j),
                                 nb_teammate_in_front(p_i_, sender),
                                 nb_opponent_in_front(p_i_, sender),
                                 compute_distance_second_closest_opponent_(p_i_, sender),
                                 compute_distance_second_closest_opponent_(p_i_, player_j)]

            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            idx += 1

    return X_pairs, y_pairs


def unmake_pair_of_player(y):
    res = np.zeros(int(len(y) / N_PLAYERS))
    for i in range(len(res)):
        game_players = y[i * N_PLAYERS:(i + 1) * N_PLAYERS] # select the N_PLAYERS players of the current game
        max_indices = np.argwhere(game_players == np.amax(game_players)).ravel() # find those that have the higher probability
        index = np.random.choice(max_indices, 1) # select one of them at random
        res[i] = index + 1 # +1 to convert index to player ID

    return res.astype('int32')


def compute_distance_(X, sender, player_j):
    d = np.sqrt((X["x_{:0.0f}".format(sender)] - X["x_{:0.0f}".format(player_j)]) ** 2
                + (X["y_{:0.0f}".format(sender)] - X["y_{:0.0f}".format(player_j)]) ** 2)
    return d


def compute_manhattan_distance_(X, sender, player_j):
    d = abs(X["x_{:0.0f}".format(sender)] - X["x_{:0.0f}".format(player_j)]) \
        + abs(X["y_{:0.0f}".format(sender)] - X["y_{:0.0f}".format(player_j)]) # manhattan distance
    return d


def compute_distance_time_(X, sender, player_j):
    # Good score = distance_time close to 0 (distance_time is negative)
    # We suppose here that a good score would be achieve with a short pass and that as time passes the probability of
    # loosing the ball increases, the pass should thus be shorter and shorter to ensure that it does not fail
    if sender == player_j:
        return -MAX_DIST # to strongly penalize the chance of having a pass from the sender to himself

    d = np.sqrt((X["x_{:0.0f}".format(sender)] - X["x_{:0.0f}".format(player_j)]) ** 2
                + (X["y_{:0.0f}".format(sender)] - X["y_{:0.0f}".format(player_j)]) ** 2)
    # np.log(X["time_start"] / MAX_TIME) included in approximately [-1.75, -0.03]
    distance_time = d / np.log(X["time_start"] / MAX_TIME)
    return distance_time


def compute_x_direction_(X, sender, player_j):
    return X["x_{:0.0f}".format(sender)] - X["x_{:0.0f}".format(player_j)]


def compute_y_direction_(X, sender, player_j):
    return X["y_{:0.0f}".format(sender)] - X["y_{:0.0f}".format(player_j)]


def compute_1D_direction(x_origin, x_destination):
    diff = x_destination - x_origin

    if abs(diff) > sys.float_info.epsilon: # to prevent divisions by 0
        diff = diff / abs(diff)

    return diff


def compute_n_opponents_in_between_(X, sender, player_j):
    # High chance to do a pass = low number of opponents between sender and player_j

    if player_j == sender or not same_team_(player_j, sender):
        # strongly penalize a pass between the sender and himself or between the sender and an opponent
        return int(N_PLAYERS / 2)

    n_opponents_in_between = 0

    players = np.arange(1, N_PLAYERS + 1)

    sender_pos = np.array([X["x_{:0.0f}".format(sender)], X["y_{:0.0f}".format(sender)]])
    player_j_pos = np.array([X["x_{:0.0f}".format(player_j)], X["y_{:0.0f}".format(player_j)]])
    diff_x_y = abs(sender_pos[0] - player_j_pos[0]) - abs(sender_pos[1] - player_j_pos[1])

    for player_i in players:
        if same_team_(player_i, player_j):
            # we will count the number of opponents of player_j
            continue

        player_i_pos = np.array([X["x_{:0.0f}".format(player_i)], X["y_{:0.0f}".format(player_i)]])
        if np.linalg.norm(sender_pos - player_i_pos) > CLOSE_DIST \
                and np.linalg.norm(player_j_pos - player_i_pos) > CLOSE_DIST:
            # we will only take the close opponents of the sender or the player_j because they are more
            # passes that are intercepted on the 2 extremities of the passes than on the middle of the pass
            # as every opponent tends to follow one of the player of the other team
            continue

        dir_from_player_j = 0
        dir_from_sender = 0
        if diff_x_y > 0: # if the sender is further from player_j on the x-axis
            dir_from_player_j = compute_1D_direction(player_j_pos[0], player_i_pos[0])
            dir_from_sender = compute_1D_direction(sender_pos[0], player_i_pos[0])
        else: # if the sender is further from player_j on the y-axis
            dir_from_player_j = compute_1D_direction(player_j_pos[1], player_i_pos[1])
            dir_from_sender = compute_1D_direction(sender_pos[1], player_i_pos[1])

        if dir_from_player_j * dir_from_sender < 0: # if the opponent is between sender and player_j
            n_opponents_in_between += 1

    return n_opponents_in_between


def compute_n_opponents_in_between_2_(X, sender, player_j):
    # High chance to do a pass = low number of opponents between sender and player_j
    if player_j == sender or not same_team_(player_j, sender):
        # strongly penalize a pass between the sender and himself or between the sender and an opponent
        return int(N_PLAYERS / 2)

    n_opponents_in_between = 0

    sender_pos = np.array([X["x_{:0.0f}".format(sender)], X["y_{:0.0f}".format(sender)]])
    player_j_pos = np.array([X["x_{:0.0f}".format(player_j)], X["y_{:0.0f}".format(player_j)]])
    avg_x = (sender_pos[0] + player_j_pos[1])/2
    avg_y = (sender_pos[1] + player_j_pos[1])/2
    avg_pos = np.array([avg_x, avg_y])
    dist_sender_player_j = np.linalg.norm(sender_pos - player_j_pos)

    players = np.arange(1, N_PLAYERS + 1)
    for player_i in players:
        if same_team_(player_i, player_j):
            # we will count the number of opponents of player_j
            continue

        player_i_pos = np.array([X["x_{:0.0f}".format(player_i)], X["y_{:0.0f}".format(player_i)]])
        dist_player_i = np.linalg.norm(avg_pos - player_i_pos)

        if dist_player_i < 1.2 * dist_sender_player_j / 2:
            # if the distance from the avg coordinates to the opponent player_i is included in a circle of 120% of the
            # radius of the circle in which player_j and the sender delimit the diameter
            n_opponents_in_between += 1

    return n_opponents_in_between


def compute_n_close_opponents_(X, player_j):
    n_close_opponents = 0

    x_player_j = X["x_{:0.0f}".format(player_j)]
    y_player_j = X["x_{:0.0f}".format(player_j)]

    players = np.arange(1, N_PLAYERS + 1)
    for player_i in players:
        if same_team_(player_i, player_j):
            continue

        dist = sqrt((X["x_{:0.0f}".format(player_i)] - x_player_j) ** 2
                    + (X["y_{:0.0f}".format(player_i)] - y_player_j) ** 2)
        if dist < CLOSE_DIST:
            n_close_opponents += 1

    return n_close_opponents


def robust_mean_position(positions, outliers_proportion, n_estimations = 15):
    assert(0 <= outliers_proportion <= 1)
    nb_good_estimations = round((1 - outliers_proportion) * len(positions))
    position_means = np.zeros((n_estimations, len(positions[0])))
    for i in range(n_estimations):
        positions_random = positions[np.random.choice(positions.shape[0], nb_good_estimations), :]
        position_means[i, :] = np.mean(positions_random, axis = 0)

    return np.mean(position_means, axis = 0)


def compute_dist_to_bulk_opponents_score(X, sender, player_j):
    if player_j == sender:
        return -2 * MAX_DIST # lowest score possible

    opponents_positions = np.zeros((int(N_PLAYERS / 2), 2))
    players = np.arange(1, N_PLAYERS + 1)
    i = 0
    for player_i in players:
        if not same_team_(player_i, player_j):
            # save the opponents player_i's of the potential receiver player_j
            opponents_positions[i, 0] = X["x_{:0.0f}".format(player_i)]
            opponents_positions[i, 1] = X["y_{:0.0f}".format(player_i)]
            i += 1

    mean_opponent_position = robust_mean_position(opponents_positions, outliers_proportion = 0.50)
    # 50% of players are in a useful position at the moment considered,
    # the others are waiting for the ball to come back in their region of the field

    sender_position = np.array([X["x_{:0.0f}".format(sender)], X["y_{:0.0f}".format(sender)]])
    player_j_position = np.array([X["x_{:0.0f}".format(player_j)], X["y_{:0.0f}".format(player_j)]])

    # the distance between the bulk of dangerous opponents and the potential receiver player_j
    dist_player_j = np.linalg.norm(mean_opponent_position - player_j_position)
    dist_sender_player_j = np.linalg.norm(sender_position - player_j_position)

    # We will assume that the direction from the sender to the bulk of opponents is the direction to the goal
    # It seems quite probable as most often, most opponents are closer to their goal than the sender of the ball
    dir_sender_opponents = compute_x_direction_btw_sender_opponents(X, sender)
    dir_sender_player_j = compute_1D_direction(sender_position[0], player_j_position[0])
    dir = 1
    if dir_sender_opponents != dir_sender_player_j and dir_sender_player_j != 0 and dir_sender_opponents != 0:
        dir = -1 # if the potential receiver player_j is not going in the x direction of
                 # the bulk of opponents with respect to the sender x coordinate, it means that he is probably not
                 # going in attack towards the goal

    # The lower the score, the higher the chance of a pass
    # We make no assumption on the fact that the potential receiver player_j is or not in in the same team as the sender
    # The closer the sender is from the potential receiver player_j, the more probable is the pass
    # If the player_j is moving towards the goal (the goal where the sender should shoot),
    # the bigger the distance between player_j and the opponents, the more probable is the pass
    # Else,
    # the lower the distance between player_j and the opponents, the more probable is the pass
    # The score gives priority to the passes that moves towards the goal where the sender should shoot
    # But among those passes, only the safest ones (the farthest from the bulk of opponents) will have the priority
    # Among the other passes (that move towards the other goal), only the safest ones will have priority
    return dist_sender_player_j - dir * dist_player_j


def compute_x_direction_btw_sender_opponents(X, sender):
    opponents_positions = np.zeros((int(N_PLAYERS / 2), 2))
    players = np.arange(1, N_PLAYERS + 1)
    i = 0
    for player_i in players:
        if not same_team_(player_i, sender):
            opponents_positions[i, 0] = X["x_{:0.0f}".format(player_i)]
            opponents_positions[i, 1] = X["y_{:0.0f}".format(player_i)]
            i += 1

    x_sender = X["x_{:0.0f}".format(sender)]
    x_opponents_mean = np.mean(opponents_positions[:, 0])

    dir_sender_opponents = compute_1D_direction(x_sender, x_opponents_mean)

    return dir_sender_opponents


def compute_angle(A, B, C): # compute positive angle ABC in degrees
    diffAB = A - B
    diffCB = C - B
    cosine_angle = np.dot(diffAB, diffCB) / np.ceil(np.linalg.norm(diffAB) * np.linalg.norm(diffCB))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def compute_angle_score_(X, sender, player_j):
    # high chance of a pass = (pass forward +) receiver far from its closest opponent in terms of angle
    if player_j == sender:
        return -90 # lowest score possible: -90 degrees # the sender will never make a pass to himself

    if not same_team_(sender, player_j):
        # low chance to make a pass to an opponent
        return -90 # lowest score: -90 degrees

    sender_pos = np.array([X["x_{:0.0f}".format(sender)], X["y_{:0.0f}".format(sender)]])
    player_j_pos = np.array([X["x_{:0.0f}".format(player_j)], X["y_{:0.0f}".format(player_j)]])

    dir_sender_opponents = compute_x_direction_btw_sender_opponents(X, sender)
    # We will assume that the x direction from the sender to the bulk of opponents is the x direction to the goal
    # It seems quite probable as most often, most opponents are closer to their goal than the sender of the ball
    dir_sender_player_j = compute_1D_direction(sender_pos[0], player_j_pos[0])

    is_going_in_attack = True
    dir_attack = 1
    if dir_sender_opponents != dir_sender_player_j:
        # if the pass is not going towards the goal where the sender should shoot to
        is_going_in_attack = False
        dir_attack = -1

    dist_sender_player_j = np.linalg.norm(sender_pos - player_j_pos)
    min_theta = 180 # theta is included in the range [0, 180]
    has_found_close_player = False

    players = np.arange(1, N_PLAYERS + 1)
    for player_i in players:
        if same_team_(player_i, player_j):
            # only opponents are dangerous
            continue

        if player_i == sender: #or player_i == player_j: # we only consider the players that are different
            # from the 2 considered for the pass (player_j and sender)
            continue

        player_i_pos = np.array([X["x_{:0.0f}".format(player_i)], X["y_{:0.0f}".format(player_i)]])
        dir_sender_player_i = compute_1D_direction(sender_pos[0], player_i_pos[0])
        if dir_sender_opponents != dir_sender_player_i * dir_attack:
            # ensures that theta is in the range [0, 180]
            # If player_j is going towards the goal where the sender should shoot,
            # only players that are located between the sender and the goal have a chance
            # to catch the ball before player_j
            # If player_j is directed towards the other way,
            # only players that are located between the sender and the other goal have a chance
            # to catch the ball before player_j
            continue

        dist_sender_player_i = np.linalg.norm(sender_pos - player_i_pos)
        if dist_sender_player_i > dist_sender_player_j:
            # only players that are at most as far as player_j is from the sender, are likely to catch the ball before player_j
            continue

        theta = compute_angle(player_i_pos, sender_pos, player_j_pos)
        # the bigger the angle, the higher the chance that the pass will go to player_j
        if theta < min_theta:
            min_theta = theta
            # We save the most dangerous angle (ie: such a small angle means another player could intercept the ball)
            # corresponding to the closest players to the player j in terms of angle
            has_found_close_player = True

    if not has_found_close_player: # no other players close to player j
        if is_going_in_attack: # the pass is very safe
            return 90 # best score = highest angle
        return 0 # if it is a pass backwards, it is safe but not very useful to reach the goal. So, it has a small chance to be played

    if is_going_in_attack: # if it is a pass forward:
        # the score will be high if the closest player in terms of angle is far enough (high theta)
        return min_theta # score is included in ]0, 180], most often in ]0, 90]

    projection_pos = np.array([sender_pos[0], player_j_pos[1]])
    # phi is included in the range [0, 90]
    if projection_pos[1] == sender_pos[1]:
        # to prevent errors
        phi = 90  # the sender and and player_j are on the same y coordinate
    else:
        phi = compute_angle(projection_pos, sender_pos, player_j_pos)
        # the bigger the angle, the closer the direction of the pass to the goal

    # if it is a pass backwards:
    # the score will be high if the closest player in terms of angle is far enough (high theta)
    # and if the ball is not moving too much backwards (low phi)
    # A too high theta would not be necessary (but yet not forbidden) as a pass backwards is already very safe
    # A very low phi is on the contrary a good point as we are not moving too much backwards
    # A high theta could compensate for a high phi in the case of safe retreat
    # when there are too many opponents in front of the sender but not behind of the sender
    return min(-phi + min_theta, 20) # score is included in ]-90, 20] instead of ]-90, 90[


def compute_region(X, player_j):
    '''
    *----------------------------*
    |   7          4          1  |
    |                            |
    |                            |
    |   8          5          2  |
    |                            |
    |                            |
    |   9          6          3  |
    *----------------------------*
    '''
    x_player_j = X["x_{:0.0f}".format(player_j)]
    y_player_j = X["y_{:0.0f}".format(player_j)]
    region = 0

    if x_player_j > MAX_X * 1 / 3:
        region = 1

    elif x_player_j < -MAX_X * 1 / 3:
        region = 7

    else:
        region = 4

    if y_player_j < MAX_Y * 1 / 3 and y_player_j > -MAX_Y * 1 / 3:
        region += 1

    elif y_player_j < -MAX_X * 1 / 3:
        region += 2

    return region


def remove_outoflimits_sender(X_pairs, y_pairs):
    indexNames_x = X_pairs[abs(X_pairs['x_sender']) > MAX_X].index
    X_pairs.drop(indexNames_x, inplace = True)
    y_pairs.drop(indexNames_x, inplace = True)

    indexNames_y = X_pairs[abs(X_pairs['y_sender']) > MAX_Y].index
    X_pairs.drop(indexNames_y, inplace = True)
    y_pairs.drop(indexNames_y, inplace = True)

    return X_pairs, y_pairs


def remove_self_pass(X_pairs, y_pairs):
    indexNames = X_pairs[X_pairs['sender'] == X_pairs['player_j']].index

    X_pairs.drop(indexNames, inplace = True)
    y_pairs.drop(indexNames, inplace = True)

    return X_pairs, y_pairs


def compute_distance_closest_opponent_(X_, player_j):
    players = np.arange(1, 23)
    min_dist = 100000.

    for player_i in players:
        if same_team_(player_i, player_j):
            continue
        dist = sqrt((X_["x_{:0.0f}".format(player_i)] - X_["x_{:0.0f}".format(player_j)]) ** 2 + (
                    X_["y_{:0.0f}".format(player_i)] - X_["x_{:0.0f}".format(player_j)]) ** 2)
        if dist < min_dist:
            min_dist = dist

    return min_dist


def compute_distance_second_closest_opponent_(X_, player_j):
    players = np.arange(1, 23)
    min_dist = 100000.
    second_min_dist = 100000.

    for player_i in players:
        if same_team_(player_i, player_j):
            continue
        dist = sqrt((X_["x_{:0.0f}".format(player_i)] - X_["x_{:0.0f}".format(player_j)]) ** 2 + (
                    X_["y_{:0.0f}".format(player_i)] - X_["x_{:0.0f}".format(player_j)]) ** 2)
        if dist < min_dist:
            second_min_dist = min_dist
            min_dist = dist

    return second_min_dist


def compute_distance_opponent_pass_(X_, sender, player_j):
    players = np.arange(1, 23)
    min_dist = 100000.
    x1 = X_["x_{:0.0f}".format(sender)]
    y1 = X_["y_{:0.0f}".format(sender)]
    x2 = X_["x_{:0.0f}".format(player_j)]
    y2 = X_["y_{:0.0f}".format(player_j)]

    if sender == player_j:
        min_dist = compute_distance_closest_opponent_(X_, sender)
    else:
        for player_i in players:
            if same_team_(sender, player_i):
                continue
            x0 = X_["x_{:0.0f}".format(player_i)]
            y0 = X_["y_{:0.0f}".format(player_i)]
            dist = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist < min_dist:
                min_dist = dist

    return min_dist


def nb_teammate_in_front(X, sender):
    players = np.arange(1, 23)
    x_sender = X["x_{:0.0f}".format(sender)]

    res = 0

    for player_i in players:

        x_player_i = X["x_{:0.0f}".format(player_i)]

        if not same_team_(sender, player_i):
            continue

        if x_player_i > x_sender:
            res += 1

    return res


def nb_opponent_in_front(X, sender):
    players = np.arange(1, 23)
    x_sender = X["x_{:0.0f}".format(sender)]

    res = 0

    for player_i in players:

        x_player_i = X["x_{:0.0f}".format(player_i)]

        if same_team_(sender, player_i):
            continue

        if x_player_i > x_sender:
            res += 1

    return res


def write_submission(predictions=None, probas=None, estimated_score=0, file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    predictions: array [n_predictions, 1]
        `predictions[i]` is the prediction for player
        receiving pass `i` (or indexes[i] if given).
    probas: array [n_predictions, 22]
        `probas[i,j]` is the probability that player `j` receives
        the ball with pass `i`.
    estimated_score: float [1]
        The estimated accuracy of predictions.
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    if predictions is None and probas is None:
        raise ValueError('Predictions and/or probas should be provided.')

    n_samples = 3000
    if indexes is None:
        indexes = np.arange(n_samples)

    if probas is None:
        print('Deriving probabilities from predictions.')
        probas = np.zeros((n_samples,22))
        for i in range(n_samples):
            probas[i, predictions[i]-1] = 1

    if predictions is None:
        print('Deriving predictions from probabilities')
        predictions = np.zeros((n_samples, ))
        for i in range(n_samples):
            mask = probas[i] == np.max(probas[i])
            selected_players = np.arange(1,23)[mask]
            predictions[i] = int(selected_players[0])


    # Writing into the file
    with open(file_name, 'w') as handle:
        # Creating header
        header = '"Id","Predicted",'
        for j in range(1,23):
            header = header + '"P_{:0.0f}",'.format(j)
        handle.write(header[:-1]+"\n")

        # Adding your estimated score
        first_line = '"Estimation",{},'.format(estimated_score)
        for j in range(1,23):
            first_line = first_line + '0,'
        handle.write(first_line[:-1]+"\n")

        # Adding your predictions
        for i in range(n_samples):
            line = "{},{:0.0f},".format(indexes[i], predictions[i])
            pj = probas[i, :]
            for j in range(22):
                line = line + '{},'.format(pj[j])
            handle.write(line[:-1]+"\n")

    return file_name


