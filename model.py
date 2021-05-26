from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from utils import *

FEATURES = ["same_team", "region_player_j",
            "time_start", "distance", "manhattan_distance", "distance_time", "x_direction", "y_direction",
            "n_opponents_in_between", "n_opponents_in_between2", "n_close_opponents",
            "dist_to_bulk_opponents_score", "angle_score", "distance_opponent_pass",
            "distance_closest_opponent_player_j", "nb_teammate_in_front", "nb_opponents_in_front"]


if __name__ == '__main__':
    # ------------------------------- Learning ------------------------------- #
    # Loading the training and testing datasets
    X_LS, y_LS, X_TS, y_TS = load_training_and_testing_data()
    X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    X_TS_pairs, y_TS_pairs = make_pair_of_players(X_TS, y_TS)
    X_all_pairs = pd.concat([X_LS_pairs, X_TS_pairs])
    y_all_pairs = pd.concat([y_LS_pairs, y_TS_pairs])


    # -------------- TRAINING --------------
    print("-------------- TRAINING --------------")

    X_LS_features = X_LS_pairs[FEATURES]
    print(FEATURES)

    # Build the model
    #scorer = make_scorer(balanced_accuracy_score)
    #param_grid = {'n_estimators': [300], 'min_samples_split': np.arange(0.01, 0.11, 0.01)}
    #grid = GridSearchCV(RandomForestClassifier(), param_grid, cv = 5, scoring = scorer)
    model = AdaBoostClassifier(base_estimator = RandomForestClassifier(n_estimators = 200, max_depth = 15, n_jobs = -1),
                               n_estimators = 3, random_state = 2)
    #model = RandomForestClassifier(n_estimators = 300, n_jobs = -1, random_state = 0)

    with measure_time('Training'):
        print('Training...')
        model.fit(X_LS_features, y_LS_pairs["pass"].ravel())
        #grid.fit(X_LS_features, y_LS_pairs["pass"].ravel())

    #model = grid.best_estimator_
    #print(f"Best Parameters: {grid.best_params_}")

    y_pred = model.predict_proba(X_LS_features)[:, 1]
    y_true = np.array(y_LS_pairs['pass'])
    receiver_pred = unmake_pair_of_player(y_pred)
    receiver_true = unmake_pair_of_player(y_true)

    train_score = accuracy_score(receiver_true, receiver_pred)
    print(f"Accuracy Score: {train_score}")

    #-------------- TESTING --------------
    print("-------------- TESTING --------------")

    X_TS_features = X_TS_pairs[FEATURES]
    y_pred = model.predict_proba(X_TS_features)[:, 1]
    y_true = np.array(y_TS_pairs['pass'])
    receiver_pred = unmake_pair_of_player(y_pred)
    receiver_true = unmake_pair_of_player(y_true)

    test_score = accuracy_score(receiver_true, receiver_pred)
    print(f"Accuracy Score: {test_score}")

    # --------------SUBMITTING---------------
    #
    # X_all_features = X_all_pairs[FEATURES]
    # model.fit(X_all_features, y_all_pairs["pass"].ravel())
    #
    # prefix = ''
    # X_submit = load_from_csv(prefix+'input_test_set.csv')
    # X_submit_pairs, _ = make_pair_of_players(X_submit)
    #
    # X_submit_features = X_submit_pairs[FEATURES]
    #
    # y_submit = model.predict_proba(X_submit_features)[:,1]
    # probas = y_submit.reshape(X_submit.shape[0], 22)
    #
    # fname = write_submission(probas=probas, estimated_score=test_score)
    # print('Submission file "{}" successfully written'.format(fname))
