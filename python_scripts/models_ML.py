import joblib
from typing import Any, Dict, List, Tuple, Type, Union, Optional
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.utils import all_estimators
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.neighbors import NearestNeighbors
from Levenshtein import distance as lev_dist
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
# Импорт метрик для оценки модели (scores)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, r2_score,
    d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
    adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
    jaccard_score, v_measure_score, brier_score_loss, d2_tweedie_score,
    cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
    average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
    top_k_accuracy_score, calinski_harabasz_score, roc_auc_score, davies_bouldin_score
)
# Импорт метрик ошибок (errors)
from sklearn.metrics import (
    max_error, mean_absolute_percentage_error, median_absolute_error,
    mean_squared_log_error, mean_squared_error, mean_absolute_error
)
import sklearn

scores = (
    accuracy_score, precision_score, recall_score, f1_score, r2_score,
    d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
    adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
    jaccard_score, v_measure_score, brier_score_loss, d2_tweedie_score,
    cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
    average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
    top_k_accuracy_score, calinski_harabasz_score, roc_auc_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score, v_measure_score, adjusted_rand_score
)

errors = (
    max_error, mean_absolute_percentage_error, median_absolute_error,
    mean_squared_log_error, mean_squared_error, mean_absolute_error
)


class Model:
    def __init__(self, model: Type[BaseEstimator] = None):
        """
        Initialize the Model instance.

        Parameters:
        model (Type[BaseEstimator], optional): The model to use. Defaults to None.
        """
        all_models = self.__check_model_type()

        if model:
            assert isinstance(model, tuple(all_models)), ('Incorrect input model type. '
                                                          f'Should be one of {type(self)} models from sklearn')
        self.__model: BaseEstimator = model  # Приватизируем атрибут model

    @property
    def model(self):
        return self.__model

    def __check_model_type(self):
        self.__model_types_with_names = all_estimators(type_filter=type(self).__name__.lower())
        all_models = [t[1] for t in self.__model_types_with_names]
        return all_models

    def fit(self, X: Any, y: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Fit the model to the data.

        Parameters:
        X (Any): Training data.
        y (Any, optional): Target values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.
        """
        assert self.__model is not None, "Model is not defined."
        self.__model.fit(X, y, *args, **kwargs)

    def predict(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Predict using the model.

        Parameters:
        X (Any): Data to predict.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        Any: Predicted values.
        """
        assert self.__model is not None, "Model is not defined."
        return self.__model.predict(X, *args, **kwargs)

    def predict_proba(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Predict class probabilities using the model.

        Parameters:
        X (Any): Data to predict.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        Any: Predicted class probabilities.
        """
        classifier_models = [t[1] for t in all_estimators(type_filter='classifier')]
        assert isinstance(self.__model, tuple(classifier_models)), ('Incorrect model type for predict_proba. '
                                                                    f'Should be one of {classifier_models}')
        return self.__model.predict_proba(X, *args, **kwargs)

    def save_model(self, path: str, *args: Any, **kwargs: Any) -> None:
        """
        Save the model to a file.

        Parameters:
        path (str): The path to save the model.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.
        """
        assert self.__model is not None, "Model is not defined."
        joblib.dump(self, path, *args, **kwargs)  # Сохраняем текущий объект Model

    @classmethod
    def load_model(cls, path: str, *args: Any, **kwargs: Any) -> 'Model':
        """
        Load a model from a file.

        Parameters:
        path (str): The path to load the model from.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        Model: An instance of the Model class with the loaded model.
        """
        try:
            model_instance = joblib.load(path, *args, **kwargs)
            assert isinstance(model_instance, cls), "Loaded object is not an instance of the expected class."
            return model_instance
        except:
            raise ValueError("You're tying to load incorrect model")

    def fit_all(self, X: Any, y: Any = None) -> Tuple[
        Dict[str, 'Model'], Dict[str, Exception]]:
        """
        Fit all available models to the data.

        Parameters:
        X (Any): Training data.
        y (Any, optional): Target values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        Tuple[Dict[str, Model], Dict[str, Exception]]: A tuple containing a dictionary of fitted models and a dictionary of errors.
        """
        fitted_models: Dict[str, 'Model'] = {}
        error_fitting: Dict[str, Exception] = {}

        for model_name, model_type in self.__model_types_with_names:
            try:
                model_instance = model_type()
                model_instance.fit(X, y)
                wrapped_model = self.__class__(model_instance)
                fitted_models[model_name] = wrapped_model
            except Exception as e:
                error_fitting[model_name] = e

        return fitted_models, error_fitting

    def get_params(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Get parameters of the model.

        Parameters:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        Dict[str, Any]: Model parameters.
        """
        assert self.__model is not None, "Model is not defined."
        return self.__model.get_params(*args, **kwargs)

    def report(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Generate a report of all metrics.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        Dict[str, float]: Dictionary with metric names as keys and their values.
        """
        assert type(self).__name__ in ('Regressor', 'Classifier'), 'Only suport (Regressor, Classifier)'

        metrics = [func for func in type(self).__dict__ if not func.startswith("__") and func not in self.stop_methods]

        report = {}
        for metric in metrics:
            method = getattr(self, metric)
            try:
                result = method(y_true, y_pred)
                report[metric] = result
            except Exception as e:
                print(f"Error calculating {metric}: {e}")

        return report


class Regressor(Model):
    def __init__(self, model: Type[BaseEstimator] = None) -> None:
        """
        Initialize the Regressor instance.

        Parameters:
        model (Type[BaseEstimator], optional): The model to use. Defaults to None.
        """
        super().__init__(model)
        self.stop_methods = ['func']

    def r2_score(self, y_test: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the R2 score.

        Parameters:
        y_test (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: R2 score.
        """
        return r2_score(y_test, y_pred, *args, **kwargs)

    def mean_absolute_error(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the Mean Absolute Error (MAE).

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: Mean Absolute Error.
        """
        return mean_absolute_error(y_true, y_pred, *args, **kwargs)

    def mean_squared_error(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the Mean Squared Error (MSE).

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: Mean Squared Error.
        """
        return mean_squared_error(y_true, y_pred, *args, **kwargs)

    def root_mean_squared_error(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE).

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: Root Mean Squared Error.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred, *args, **kwargs))

    def max_error(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the Max Error.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: Max Error.
        """
        return max_error(y_true, y_pred, *args, **kwargs)

    def mean_absolute_percentage_error(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE).

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: Mean Absolute Percentage Error.
        """
        return mean_absolute_percentage_error(y_true, y_pred, *args, **kwargs)

    def median_absolute_error(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the Median Absolute Error (MedAE).

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: Median Absolute Error.
        """
        return median_absolute_error(y_true, y_pred, *args, **kwargs)

    def mean_squared_log_error(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the Mean Squared Logarithmic Error (MSLE).

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: Mean Squared Logarithmic Error.
        """
        return mean_squared_log_error(y_true, y_pred, *args, **kwargs)

    def d2_absolute_error_score(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the D2 Absolute Error Score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        float: D2 Absolute Error Score.
        """
        return d2_absolute_error_score(y_true, y_pred, *args, **kwargs)

    def root_mean_squared_log_error(self, y_true: Any, y_pred: Any, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the Root Mean Squared Logarithmic Error (RMSLE).

        Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

        Returns:
        float: Root Mean Squared Logarithmic Error.
        """
        return np.sqrt(mean_squared_log_error(y_true, y_pred, *args, **kwargs))

    def func(self):
        # type(self).__dict__
        return dir(self)


class Classifier(Model):
    def __init__(self, model: Type[BaseEstimator] = None) -> None:
        """
        Initialize the Classifier instance.

        Parameters:
        model (Type[BaseEstimator], optional): The model to use. Defaults to None.
        """
        super().__init__(model)
        self.stop_methods = ['roc_auc_plot', 'confusion_matrix_display']

    def accuracy_score(self, y_true: Any, y_pred: Any, *args: Any,
                       **kwargs: Any) -> float:
        """
        Calculate the accuracy score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Accuracy score.
        """
        return accuracy_score(y_true, y_pred, *args, **kwargs)

    def precision_score(self, y_true: Any, y_pred: Any, *args: Any,
                        **kwargs: Any) -> float:
        """
        Calculate the precision score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Precision score.
        """
        return precision_score(y_true, y_pred, *args, **kwargs)

    def recall_score(self, y_true: Any, y_pred: Any, *args: Any,
                     **kwargs: Any) -> float:
        """
        Calculate the recall score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Recall score.
        """
        return recall_score(y_true, y_pred, *args, **kwargs)

    def f1_score(self, y_true: Any, y_pred: Any, *args: Any,
                 **kwargs: Any) -> float:
        """
        Calculate the F1 score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: F1 score.
        """
        return f1_score(y_true, y_pred, *args, **kwargs)

    def d2_absolute_error_score(self, y_true: Any, y_pred: Any, *args: Any,
                                **kwargs: Any) -> float:
        """
        Calculate the D2 Absolute Error score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: D2 Absolute Error score.
        """
        return d2_absolute_error_score(y_true, y_pred, *args, **kwargs)

    def ndcg_score(self, y_true: Any, y_pred: Any, *args: Any,
                   **kwargs: Any) -> float:
        """
        Calculate the NDCG score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: NDCG score.
        """
        return ndcg_score(y_true, y_pred, *args, **kwargs)

    def rand_score(self, y_true: Any, y_pred: Any, *args: Any,
                   **kwargs: Any) -> float:
        """
        Calculate the Rand score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Rand score.
        """
        return rand_score(y_true, y_pred, *args, **kwargs)

    def dcg_score(self, y_true: Any, y_pred: Any, *args: Any,
                  **kwargs: Any) -> float:
        """
        Calculate the DCG score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: DCG score.
        """
        return dcg_score(y_true, y_pred, *args, **kwargs)

    def fbeta_score(self, y_true: Any, y_pred: Any, beta: float, *args: Any,
                    **kwargs: Any) -> float:
        """
        Calculate the F-beta score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        beta (float): Weight of precision in harmonic mean.

        Returns:
        float: F-beta score.
        """
        return fbeta_score(y_true, y_pred, beta, *args, **kwargs)

    def adjusted_rand_score(self, y_true: Any, y_pred: Any, *args: Any,
                            **kwargs: Any) -> float:
        """
        Calculate the Adjusted Rand score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Adjusted Rand score.
        """
        return adjusted_rand_score(y_true, y_pred, *args, **kwargs)

    def silhouette_score(self, X: Any, labels: Any, *args: Any,
                         **kwargs: Any) -> float:
        """
        Calculate the Silhouette score.

        Parameters:
        X (Any): Feature data.
        labels (Any): Cluster labels.

        Returns:
        float: Silhouette score.
        """
        return silhouette_score(X, labels, *args, **kwargs)

    def completeness_score(self, y_true: Any, y_pred: Any, *args: Any,
                           **kwargs: Any) -> float:
        """
        Calculate the Completeness score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Completeness score.
        """
        return completeness_score(y_true, y_pred, *args, **kwargs)

    def homogeneity_score(self, y_true: Any, y_pred: Any, *args: Any,
                          **kwargs: Any) -> float:
        """
        Calculate the Homogeneity score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Homogeneity score.
        """
        return homogeneity_score(y_true, y_pred, *args, **kwargs)

    def jaccard_score(self, y_true: Any, y_pred: Any, *args: Any,
                      **kwargs: Any) -> float:
        """
        Calculate the Jaccard score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Jaccard score.
        """
        return jaccard_score(y_true, y_pred, *args, **kwargs)

    def v_measure_score(self, y_true: Any, y_pred: Any, *args: Any,
                        **kwargs: Any) -> float:
        """
        Calculate the V-measure score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: V-measure score.
        """
        return v_measure_score(y_true, y_pred, *args, **kwargs)

    def brier_score_loss(self, y_true: Any, y_prob: Any, *args: Any,
                         **kwargs: Any) -> float:
        """
        Calculate the Brier score loss.

        Parameters:
        y_true (Any): True values.
        y_prob (Any): Predicted probabilities.

        Returns:
        float: Brier score loss.
        """
        return brier_score_loss(y_true, y_prob, *args, **kwargs)

    def d2_tweedie_score(self, y_true: Any, y_pred: Any, *args: Any,
                         **kwargs: Any) -> float:
        """
        Calculate the D2 Tweedie score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: D2 Tweedie score.
        """
        return d2_tweedie_score(y_true, y_pred, *args, **kwargs)

    def cohen_kappa_score(self, y1: Any, y2: Any, *args: Any,
                          **kwargs: Any) -> float:
        """
        Calculate the Cohen's kappa score.

        Parameters:
        y1 (Any): First set of labels.
        y2 (Any): Second set of labels.

        Returns:
        float: Cohen's kappa score.
        """
        return cohen_kappa_score(y1, y2, *args, **kwargs)

    def d2_pinball_score(self, y_true: Any, y_pred: Any, *args: Any,
                         **kwargs: Any) -> float:
        """
        Calculate the D2 Pinball score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: D2 Pinball score.
        """
        return d2_pinball_score(y_true, y_pred, *args, **kwargs)

    def mutual_info_score(self, y_true: Any, y_pred: Any, *args: Any,
                          **kwargs: Any) -> float:
        """
        Calculate the Mutual Information score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Mutual Information score.
        """
        return mutual_info_score(y_true, y_pred, *args, **kwargs)

    def adjusted_mutual_info_score(self, y_true: Any, y_pred: Any, *args: Any,
                                   **kwargs: Any) -> float:
        """
        Calculate the Adjusted Mutual Information score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Adjusted Mutual Information score.
        """
        return adjusted_mutual_info_score(y_true, y_pred, *args, **kwargs)

    def average_precision_score(self, y_true: Any, y_score: Any, *args: Any,
                                **kwargs: Any) -> float:
        """
        Calculate the Average Precision score.

        Parameters:
        y_true (Any): True values.
        y_score (Any): Predicted scores/probabilities.

        Returns:
        float: Average Precision score.
        """
        return average_precision_score(y_true, y_score, *args, **kwargs)

    def label_ranking_average_precision_score(self, y_true: Any, y_score: Any, *args: Any,
                                              **kwargs: Any) -> float:
        """
        Calculate the Label Ranking Average Precision score.

        Parameters:
        y_true (Any): True values.
        y_score (Any): Predicted scores/probabilities.

        Returns:
        float: Label Ranking Average Precision score.
        """
        return label_ranking_average_precision_score(y_true, y_score, *args, **kwargs)

    def balanced_accuracy_score(self, y_true: Any, y_pred: Any, *args: Any,
                                **kwargs: Any) -> float:
        """
        Calculate the Balanced Accuracy score.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        float: Balanced Accuracy score.
        """
        return balanced_accuracy_score(y_true, y_pred, *args, **kwargs)

    def top_k_accuracy_score(self, y_true: Any, y_score: Any, k: int, *args: Any,
                             **kwargs: Any) -> float:
        """
        Calculate the Top-K Accuracy score.

        Parameters:
        y_true (Any): True values.
        y_score (Any): Predicted scores/probabilities.
        k (int): Number of top elements to consider.

        Returns:
        float: Top-K Accuracy score.
        """
        return top_k_accuracy_score(y_true, y_score, k, *args, **kwargs)

    def calinski_harabasz_score(self, X: Any, labels: Any, *args: Any,
                                **kwargs: Any) -> float:
        """
        Calculate the Calinski-Harabasz score.

        Parameters:
        X (Any): Feature data.
        labels (Any): Cluster labels.

        Returns:
        float: Calinski-Harabasz score.
        """
        return calinski_harabasz_score(X, labels, *args, **kwargs)

    def roc_auc_score(self, y_true: Any, y_score: Any, *args: Any,
                      **kwargs: Any) -> float:
        """
        Calculate the ROC AUC score.

        Parameters:
        y_true (Any): True values.
        y_score (Any): Predicted scores/probabilities.

        Returns:
        float: ROC AUC score.
        """
        return roc_auc_score(y_true, y_score, *args, **kwargs)

    def roc_auc_plot(self, y_true: Any, y_score: Any, *args: Any,
                     **kwargs: Any) -> None:
        """
        Plot the ROC curve.

        Parameters:
        y_true (array-like): True values.
        y_score (array-like): Predicted scores.

        Returns:
        None
        """
        fpr, tpr, _ = roc_curve(y_true, y_score, *args, **kwargs)
        roc_auc = self.roc_auc_score(y_true, y_score)

        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()

    def confusion_matrix_display(self, x_test: Any, y_test: Any, *args: Any,
                                 **kwargs: Any) -> ConfusionMatrixDisplay:
        """
        Display the Confusion Matrix.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        ConfusionMatrixDisplay: Confusion Matrix display object.
        """
        return ConfusionMatrixDisplay.from_estimator(self.model, x_test, y_test, *args, **kwargs)


class Cluster(Model):
    def __init__(self, model: Type[BaseEstimator] = None) -> None:
        """
        Initialize the Cluster instance.

        Parameters:
        model (Type[BaseEstimator], optional): The model to use. Defaults to None.
        """
        super().__init__(model)
        self.stop_methods = ['elbow_method', 'elbow_method_plot', 'elbow_method_best_k']

    @property
    def labels_(self):
        return self.model.labels_

    @property
    def n_clusters(self):
        return self.model.n_clusters

    def elbow_method(self, x_train: Any, max_k: int, change_n_clusters: bool = True) -> List[float]:
        """
        Apply the elbow method to determine the optimal number of clusters and optionally update the model.

        Parameters:
        x_train (array-like): Training data.
        max_k (int): Maximum number of clusters to consider.
        change_n_clusters (bool, optional): If True, update the model's 'n_clusters' parameter to the optimal number and fit the model. Defaults to True.

        Returns:
        list: WCSS (within-cluster sum of squares) for each number of clusters.
        """
        assert isinstance(max_k, int), f'Incorrect max_k param type. {type(max_k)} instead of {int}'
        assert self.model.__class__.__name__ in ('BisectingKMeans', 'KMeans', 'MiniBatchKMeans'), \
            f"This model doesn't support the elbow method. Valid models: {('BisectingKMeans', 'KMeans', 'MiniBatchKMeans')}"

        default_num_clusters = self.model.n_clusters

        wcss = []
        for k in range(1, max_k + 1):
            self.model.n_clusters = k
            model = self.model.fit(x_train)
            wcss.append(model.inertia_)

        n_clust = self.__elbow_method_best_k(wcss)
        if change_n_clusters:
            self.model.n_clusters = n_clust
            self.model.fit(x_train)
            print(f"Your model's parameter 'n_clusters' was changed to optimal: {n_clust} and model was fitted on it.")
        else:
            self.model.n_clusters = default_num_clusters

        return wcss

    def elbow_method_plot(self, wcss: Union[List[float], Tuple[float, ...]]) -> None:
        """
        Plot the results of the elbow method.

        Parameters:
        wcss (list or tuple): WCSS values for different numbers of clusters.

        Returns:
        None
        """
        assert isinstance(wcss, (list, tuple)), f'Incorrect wcss param type. {type(wcss)} instead of {list | tuple}'

        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.plot(range(1, len(wcss) + 1), wcss, marker='o', mfc='red')
        plt.title('Selecting the number of clusters using the elbow method')
        plt.xlabel('num clusters')
        plt.ylabel('WCSS (error)')
        plt.xticks(range(1, len(wcss) + 1))
        plt.show()

    def __elbow_method_best_k(self, wcss: Union[List[float], Tuple[float, ...]]) -> Union[int, str]:
        """
        Determine the best number of clusters using the elbow method with a given threshold.

        Parameters:
        wcss (list or tuple): WCSS values for different numbers of clusters.

        Returns:
        int: Optimal number of clusters.
        """
        assert isinstance(wcss, (list, tuple)), f'Incorrect wcss parameter type. {type(wcss)} instead of {list | tuple}'
        assert len(wcss) >= 3, 'max_k len must be >= 3'

        # подробное описание работы алгоритма в файле про кластеризацию и метрики качества
        diff = np.diff(wcss)
        diff_r = diff[1:] / diff[:-1]
        k_opt = range(1, len(wcss))[np.argmin(diff_r) + 1]

        return k_opt

    def silhouette_score(self, x_train: Any, labels: Any) -> float:
        """
        Calculate the silhouette score for the given data and labels.

        Parameters:
        x_train (array-like): Training data.
        labels (array-like): Cluster labels.

        Returns:
        float: Silhouette score.
        """
        return silhouette_score(x_train, labels)

    def calinski_harabasz_score(self, x_train: Any, labels: Any) -> float:
        """
        Calculate the Calinski-Harabasz index for the given data and labels.

        Parameters:
        x_train (array-like): Training data.
        labels (array-like): Cluster labels.

        Returns:
        float: Calinski-Harabasz index.
        """
        return calinski_harabasz_score(x_train, labels)

    def davies_bouldin_score(self, x_train: Any, labels: Any) -> float:
        """
        Calculate the Davies-Bouldin index for the given data and labels.

        Parameters:
        x_train (array-like): Training data.
        labels (array-like): Cluster labels.

        Returns:
        float: Davies-Bouldin index.
        """
        return davies_bouldin_score(x_train, labels)

    def v_measure_score(self, y_true: Any, y_pred: Any) -> float:
        """
        Calculate the V-measure for the given true labels and predicted labels.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

        Returns:
        float: V-measure score.
        """
        return v_measure_score(y_true, y_pred)

    def adjusted_rand_index(self, y_true: Any, y_pred: Any) -> float:
        """
        Calculate the Adjusted Rand Index (ARI) for the given true labels and predicted labels.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

        Returns:
        float: Adjusted Rand Index.
        """
        return adjusted_rand_score(y_true, y_pred)

    def dunn_index(self, x_train: Any, labels: Any) -> float:
        """
        Calculate the Dunn Index for the given data and labels.

        Parameters:
        x_train (array-like): Training data.
        labels (array-like): Cluster labels.

        Returns:
        float: Dunn Index.
        """
        clusters = np.unique(labels)
        if len(clusters) < 2:
            return 0

        distances = cdist(x_train, x_train)
        intra_cluster_dists = [np.max(distances[labels == cluster]) for cluster in clusters]
        inter_cluster_dists = [np.min(distances[labels == c1][:, labels == c2])
                               for i, c1 in enumerate(clusters) for c2 in clusters[i + 1:]]

        return np.min(inter_cluster_dists) / np.max(intra_cluster_dists)

    def report(self, x_train: Any, y_true: Any, labels: Any) -> Dict[str, float]:
        """
        Calculate all clustering metrics for the given data and labels.

        Parameters:
        x_train (array-like): Training data.
        y_true (array-like): True labels.
        labels (array-like): Cluster labels.

        Returns:
        Dict[str, float]: Dictionary with metric names as keys and their values.
        """
        metrics = {
            'Silhouette Score': self.silhouette_score(x_train, labels),
            'Calinski-Harabasz Index': self.calinski_harabasz_score(x_train, labels),
            'Davies-Bouldin Index': self.davies_bouldin_score(x_train, labels),
            'Dunn Index': self.dunn_index(x_train, labels),
            'V-Measure': self.v_measure_score(y_true, labels),
            'Adjusted Rand Index': self.adjusted_rand_index(y_true, labels)
        }
        return metrics

    def __plot_dendrogram(self, model, **kwargs):
        """
        Generate the linkage matrix and plot the dendrogram.

        Parameters:
        model (object): Fitted clustering model.
        kwargs: Additional keyword arguments for the dendrogram plotting function.

        Returns:
        None
        """

        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
        dendrogram(linkage_matrix, **kwargs)

    def dendrogram_plot(self, **kwargs):
        """
        Plot a dendrogram for the agglomerative clustering model.

        Parameters:
        kwargs: Additional keyword arguments for the dendrogram plotting function.

        Returns:
        None
        """

        assert self.model.__class__.__name__ in ('AgglomerativeClustering'), f'Only support AgglomerativeClustering'
        assert hasattr(self.model, 'children_'), f'The model must be fitted'

        plt.figure(figsize=(10, 8))
        plt.title('Hierarchical Clustering Dendrogram')
        self.__plot_dendrogram(self.model, **kwargs)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()


class RecommendSystem:
    def __init__(self, based_on: Optional[Any] = None) -> None:
        """
        Constructor to initialize the RecommendSystem class.

        Parameters:
        based_on (optional): An optional parameter that can be used to customize the initialization.
        """
        self.__based_on = based_on

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        """
        Fit the recommendation model with provided data.

        Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target vector.
        **kwargs: Additional keyword arguments for configuring the NearestNeighbors model, such as:
            - n_neighbors (int): Number of neighbors to use. Default is the number of rows in the weighted average DataFrame.
            - radius (float): Range of parameter space to use by default for neighbors search. Default is 1.0.
            - metric (str): Metric to use for distance computation. Default is "minkowski".

        Returns:
        None: This method does not return any value. It fits the model with the provided data.
        """
        df = pd.concat([X, y], axis=1)
        weighted_avg = df.groupby(df.columns[-1]).apply(
            lambda g: g.iloc[:, :-1].multiply(len(g), axis=0).sum() / len(g))
        self.df = weighted_avg

        self.__model = NearestNeighbors(
            n_neighbors=kwargs.get('n_neighbors', self.df.shape[0]),
            radius=kwargs.get('radius', 1.0),
            algorithm="auto",
            leaf_size=30,
            metric=kwargs.get('metric', "minkowski"),
            p=2,
            metric_params=None,
            n_jobs=None
        )
        self.__model.fit(self.df)

    def predict(self, x: Union[pd.DataFrame, List[Any]], **kwargs: Any) -> List[pd.DataFrame]:
        """
        Predict recommendations for the given input.

        Parameters:
        x (DataFrame or list): Input data for which recommendations are to be made. If a list is provided, it will be converted to a DataFrame.
        **kwargs: Additional keyword arguments for configuring the prediction, such as:
            - ascending (bool): Whether to sort the distances in ascending order. Default is False.

        Returns:
        list of DataFrames: Each DataFrame contains the recommendations and distances for the corresponding input.
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame([x])

        result = self.__model.kneighbors(x, return_distance=True)
        res_recomends = []
        for example, dist in zip(result[1], result[0]):
            temp_df = self.df.copy()
            temp_df['recommendation'] = example
            temp_df['distance'] = dist
            temp_df.sort_values('distance', inplace=True, ascending=kwargs.get('ascending', False))
            temp_df.reset_index(inplace=True, drop=True)
            res_recomends.append(temp_df)
        return res_recomends

    def levenshtein_distance_handmade(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.

        Parameters:
        s1 (str): First string.
        s2 (str): Second string.

        Returns:
        int: The Levenshtein distance between the two strings.
        """
        len_s1 = len(s1)
        len_s2 = len(s2)

        dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

        for i in range(len_s1 + 1):
            dp[i][0] = i

        for j in range(len_s2 + 1):
            dp[0][j] = j

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1,
                               dp[i][j - 1] + 1,
                               dp[i - 1][j - 1] + cost)

        return dp[len_s1][len_s2]

    def report(self, x: Union[pd.DataFrame, List[Any]], **kwargs: Any) -> pd.DataFrame:
        """
        Generate a report of recommendations using various distance metrics.

        Parameters:
        x (DataFrame or list): Input data for which the report is to be generated. If a list is provided, it will be converted to a DataFrame.
        **kwargs: Additional keyword arguments for configuring the report, such as:
            - n_neighbors (int): Number of neighbors to use. Default is the number of rows in the weighted average DataFrame.
            - radius (float): Range of parameter space to use by default for neighbors search. Default is 1.0.
            - sort_by (str): The distance metric to sort the recommendations by. Default is 'minkowski'.
            - ascending (bool): Whether to sort the distances in ascending order. Default is False.

        Returns:
        DataFrame: A DataFrame containing the recommendations and distances for the given input, sorted by the specified metric.
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame([x])

        recommendation_metrics = ['minkowski', 'cityblock', 'cosine', 'euclidean', 'haversine',
                                  'l1', 'l2', 'manhattan', 'nan_euclidean', lev_dist]
        temp_df = self.df.copy()
        for metric in recommendation_metrics:
            model = NearestNeighbors(
                n_neighbors=kwargs.get('n_neighbors', self.df.shape[0]),
                radius=kwargs.get('radius', 1.0),
                algorithm="auto",
                leaf_size=30,
                metric=metric,
                p=2,
                metric_params=None,
                n_jobs=None
            )
            model.fit(self.df)
            result = model.kneighbors(x, return_distance=True)
            name_metric = f'distance_{metric}' if metric != recommendation_metrics[-1] else f'distance_levenshtein'
            temp_df[name_metric] = result[0][0]
        temp_df.sort_values(f"distance_{kwargs.get('sort_by', 'minkowski')}", inplace=True,
                            ascending=kwargs.get('ascending', False))
        temp_df.reset_index(inplace=True, drop=True)
        return temp_df
