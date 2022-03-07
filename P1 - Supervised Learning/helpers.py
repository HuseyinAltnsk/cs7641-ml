import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools

def getXy(data):
    X = data.values[:,:-1]
    y = data.values[:, -1]
    return X, y

def get_splits(data, test_size):
    X, y = getXy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

# Hyperparameter Tuning with GridSearchCV
def HT_GridSearchCV(clf, splits, param_dict, cv, clf_short_name):
    
    X_train, X_test, y_train, y_test = splits
    grid = GridSearchCV(clf, param_grid=param_dict, cv=cv, verbose=1, n_jobs=-1)

    start_time = time.time() # Time the classifier
    grid.fit(X_train, y_train)
    end_time = time.time()
    train_duration = end_time - start_time
    print(f"Best {clf_short_name} - Training Duration: {train_duration:.2f}")

    start_time = time.time() # Time the classifier
    y_pred = grid.predict(X_test)
    end_time = time.time()
    pred_duration = end_time - start_time
    print(f"Best {clf_short_name} - Prediction Duration: {pred_duration:.2f}")

    best_accuracy = accuracy_score(y_test, y_pred)
    best_f1 = f1_score(y_test, y_pred)
    best_precision = precision_score(y_test, y_pred)
    best_recall = recall_score(y_test, y_pred)
    print(f"BEST {clf_short_name} SCORE RESULTS:")
    print(f"Accuracy: {best_accuracy*100:.2f}%")
    print(f"F1: {best_f1:.2f}")
    print(f"Precision: {best_precision:.2f}")
    print(f"Recall: {best_recall:.2f}")
    # print(f"Highest Mean Score: {grid.best_score_:.2f}")
    print("Best Combination of Parameters: ", grid.best_params_)

    return grid.best_estimator_, train_duration, pred_duration, best_accuracy, best_f1


def plot_learning_curve(estimator, X, y, train_sizes, cv=None, clf_long_name="Classifier", dataset="Unknown"):
    
    _, axes = plt.subplots(1, 3, figsize=(15, 4))

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)


    # Plot learning curve
    axes[0].set_title("Learning Curve for (Hyperparameter-Tuned) " + clf_long_name)
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="b", label="Training Score")
    axes[0].plot(train_sizes, test_scores_mean, "o-", color="r", label="CV Score")
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="b", alpha=0.2)
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="r", alpha=0.2)
    axes[0].set_xlabel("Fraction of Training Examples")
    axes[0].set_ylabel("Classification Score")
    axes[0].legend() # loc="best" by default
    axes[0].grid()

    # Plot n_samples vs fit_times
    axes[1].set_title(f"Scalability of the {clf_long_name}")
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.2)
    axes[1].set_xlabel("Fraction of Training Examples")
    axes[1].set_ylabel("Time Spent for Fitting (s)")
    axes[1].grid()

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].set_title(f"Performance of the {clf_long_name}")
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(fit_time_sorted, test_scores_mean_sorted - test_scores_std_sorted, test_scores_mean_sorted + test_scores_std_sorted, alpha=0.2)
    axes[2].set_xlabel("Time Spent for Scoring (s)")
    axes[2].set_ylabel("Classification Score")
    axes[2].grid()
    
    plt.savefig(dataset + " - " + clf_long_name + "_Learning_Curve")
    return plt

def plot_validation_curve(estimator, X, y, param_name, param_range, cv=None, 
        isLogScale=False, clf_long_name="Classifier", dataset="Unknown"):

    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name,
                                                param_range=param_range, cv=cv, scoring="accuracy", n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(f"Validation Curve for {clf_long_name}")
    plt.xlabel(f"Parameter: '{param_name}'")
    plt.ylabel("Classification Score")

    if isLogScale:
        plt.semilogx(param_range, train_scores_mean, 'o-', color="orange", label="Training Score")
        plt.semilogx(param_range, test_scores_mean, 'o-', color="g", label="CV Score")
    else:
        plt.plot(param_range, train_scores_mean, 'o-', color='orange', label='Training Score')
        plt.plot(param_range, test_scores_mean, 'o-', color='g', label='CV Score')
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="orange", alpha=0.2)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="g", alpha=0.2)
    plt.legend() # loc="best" by default
    plt.tight_layout()
    plt.grid()
    
    plt.savefig(dataset + " - " + clf_long_name + "_Validation_Curve")
    return plt

# plot_confusion_matrix(cm, classes=['0', '1'], clf_long_name="Decision Tree")# 0==no, 1==yes
# def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues, clf_long_name="Classifier"):

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(f"Confusion Matrix for (Hyperparameter-Tuned) {clf_long_name}")
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(2), range(2)):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.grid()
    
#     return plt