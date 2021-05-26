from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV

from utils import *

# Scores: 0.310 - 0.280 (AdaBoost n_estimators: 300)
# Scores: 0.318 - 0.289 (RandomForest n_estimators: 300 ; min_samples_split: 0.02)
FEATURES = ["same_team", "region_player_j", "region_sender",
            "time_start", "distance", "manhattan_distance", "distance_time", "x_direction", "y_direction",
            "n_opponents_in_between", "n_opponents_in_between2", "n_close_opponents",
            "dist_to_bulk_opponents_score", "angle_score", "distance_opponent_pass",
            "distance_closest_opponent_sender", "distance_closest_opponent_player_j",
            "nb_teammate_in_front", "nb_opponents_in_front"]

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
        # other_players = np.delete(players, sender-1)
        p_i_ = X_.iloc[i]
        for player_j in players:

            X_pairs.iloc[idx] = [sender, p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j),
                                 compute_x_region(p_i_, player_j),
                                 compute_y_region(p_i_, player_j),
                                 compute_x_region(p_i_, sender),
                                 compute_y_region(p_i_, sender),
                                 p_i_["time_start"],
                                 compute_distance_(p_i_, sender, player_j),
                                 compute_manhattan_distance_(p_i_, sender, player_j),
                                 compute_distance_time_(p_i_, sender, player_j),
                                 compute_x_direction_(p_i_, sender, player_j),
                                 compute_y_direction_(p_i_, sender, player_j),
                                 compute_n_opponents_in_between_(p_i_, sender, player_j),
                                 compute_n_opponents_in_between_2_(p_i_, sender, player_j),
                                 compute_n_close_opponents_(p_i_, player_j),
                                 dist_to_bulk_opponents(p_i_, sender, player_j)]

            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            idx += 1

    return X_pairs, y_pairs


if __name__ == '__main__':
    # ------------------------------- Learning ------------------------------- #
    # Loading the training and testing datasets
    #X_LS, y_LS, X_TS, y_TS = load_training_and_testing_data()

    # -------------- TRAINING --------------
    print("-------------- TRAINING --------------")

    #X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    X_LS_features = X_LS_pairs[FEATURES]
    print(FEATURES)

    # Build the model
    scorer = make_scorer(balanced_accuracy_score)
    param_grid = {'n_estimators': [300], 'min_samples_split': [0.02]}#'max_depth': np.arange(1, 30)}
    #param_grid = {'max_depth': np.arange(1, 30)}
    #grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv = 5, scoring = scorer)
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv = 5, scoring = scorer)

    with measure_time('Training'):
        print('Training...')
        grid.fit(X_LS_features, y_LS_pairs["pass"].ravel())

    model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")

    y_pred = model.predict_proba(X_LS_features)[:, 1]
    y_true = np.array(y_LS_pairs['pass'])
    receiver_pred = unmake_pair_of_player(y_pred)
    receiver_true = unmake_pair_of_player(y_true)

    train_score = balanced_accuracy_score(receiver_true, receiver_pred)
    print(f"Balanced Accuracy Score: {train_score}")


    #-------------- TESTING --------------
    print("-------------- TESTING --------------")
    #X_TS_pairs, y_TS_pairs = make_pair_of_players(X_TS, y_TS)

    X_TS_features = X_TS_pairs[FEATURES]
    y_pred = model.predict_proba(X_TS_features)[:, 1]
    y_true = np.array(y_TS_pairs['pass'])

    receiver_pred = unmake_pair_of_player(y_pred)
    receiver_true = unmake_pair_of_player(y_true)

    test_score = balanced_accuracy_score(receiver_true, receiver_pred)
    print(f"Balanced Accuracy Score: {test_score}")

    # --------------SUBMITTING---------------
    '''
    prefix = ''
    X_submit = load_from_csv(prefix+'input_test_set.csv')
    X_submit_pairs, _ = make_pair_of_players_submission(X_submit, submission = True)

    X_submit_features = X_submit_pairs[FEATURES]

    y_submit = model.predict_proba(X_submit_features)[:,1]
    probas = y_submit.reshape(X_submit.shape[0], 22)

    fname = write_submission(probas=probas, estimated_score=test_score)
    print('Submission file "{}" successfully written'.format(fname))
    '''
