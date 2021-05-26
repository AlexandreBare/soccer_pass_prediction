from test import *
import matplotlib.pyplot as plt

def remove_failed_passes(X, y):
    n_failed_passes = 0
    n_passes = len(y)
    X_res = X
    y_res = y
    for i in range(0, n_passes):
        if not same_team_(y.iloc[i].receiver, X.iloc[i].sender):
            remove = X.iloc[i].Id
            X_res = X_res.drop(index = remove)
            y_res = y_res.drop(index = remove)
            n_failed_passes += 1


    ratio_failed_passes = n_failed_passes / n_passes
    return X_res, y_res, ratio_failed_passes

if __name__ == '__main__':
    prefix = ''

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_LS = load_from_csv(prefix + 'input_training_set.csv')
    y_LS = load_from_csv(prefix + 'output_training_set.csv')

    # divide the sample size to compute quicker results
    sub_dataset_size = len(X_LS)#floor(len(X_LS)/10)
    X_LS = X_LS[0:sub_dataset_size]
    y_LS = y_LS[0:sub_dataset_size]

    plt.figure()
    for i in range(N_PLAYERS):
        plt.scatter(np.quantile(X_LS["x_{:0.0f}".format(i + 1)], 0.5), np.quantile(X_LS["y_{:0.0f}".format(i + 1)], 0.5), label = f"Player #{i + 1}", s = int((i+1) > 11) * 100 + 50)

    plt.show()

    # Train - Test Set split
    X_LS, X_TS, y_LS, y_TS = train_test_split(X_LS, y_LS, test_size = 0.5, random_state = 1)

    #X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    #X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)

    #X_LS_features = X_LS_pairs[["distance", "same_team", "sender", "player_j"]]

    print("-------------- Data Analysis --------------")

    X_res, y_res, ratio_failed_passes = remove_failed_passes(X_LS, y_LS)
    print(f"Ratio of failed passes: {ratio_failed_passes}")

    max_predict_score = 1 - ratio_failed_passes

