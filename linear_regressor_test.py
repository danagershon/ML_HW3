import unittest
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from LinearRegressor import LinearRegressor
from prepare import prepare_data
from verify_gradients import compare_gradients
from test_lr import test_lr


class TestLinearRegressor(unittest.TestCase):

    def setUp(self) -> None:
        # load train annd test sets data from csv files
        dataset = pd.read_csv('data_HW3.csv')
        df_train, df_test = train_test_split(dataset, train_size=0.8, random_state=74+40)

        # Prepare training set according to itself
        self.train_df_prepared = prepare_data(df_train, df_train)

        # Prepare test set according to the raw training set
        self.test_df_prepared = prepare_data(df_train, df_test)

        self.df_train_subset, self.df_validation_subset = train_test_split(self.train_df_prepared, train_size=0.8, random_state=74+40)
    
    def test_compare_gradients(self):
        X_train = self.df_train_subset.drop('contamination_level', axis=1).values
        y_train = self.df_train_subset['contamination_level'].values

        compare_gradients(X_train, y_train, deltas=np.logspace(-7, -2, 9))

    def test_lr(self):
        X_train = self.df_train_subset.drop('contamination_level', axis=1).values
        y_train = self.df_train_subset['contamination_level'].values

        X_val = self.df_validation_subset.drop('contamination_level', axis=1).values
        y_val = self.df_validation_subset['contamination_level'].values

        test_lr(X_train, y_train, X_val, y_val, title="Training and Validation Losses as a Function of Iteration # for Different LRs")

    def test_cv(self):
        # Q5 CV on LinearRegressor, tuning LR

        whole_X_train = self.train_df_prepared.drop('contamination_level', axis=1)
        whole_y_train = self.train_df_prepared['contamination_level']
        
        # LR range
        learning_rates = np.logspace(-6, -1, 10)

        train_errors = []
        validation_errors = []

        for lr in learning_rates:
            linear_regressor = LinearRegressor(lr=lr)
            cv_results = cross_validate(linear_regressor, whole_X_train, whole_y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
            train_errors.append(np.mean(cv_results['train_score']))
            validation_errors.append(np.mean(cv_results['test_score']))

        # Convert negative MSE to positive for readability
        train_errors = [-score for score in train_errors]
        validation_errors = [-score for score in validation_errors]

        # Plot the cross-validated train and validation errors
        plt.figure(figsize=(12, 6))
        plt.semilogx(learning_rates, train_errors, label='Train Error', marker='o')
        plt.semilogx(learning_rates, validation_errors, label='Validation Error', marker='o')
        plt.xlabel('Learning Rate')
        plt.ylabel('Mean Squared Error')
        plt.title('Linear Regressor Train and Validation Errors as a function of LR')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Find the best learning rate
        optimal_lr_idx = np.argmin(validation_errors)
        optimal_lr = learning_rates[optimal_lr_idx]
        optimal_validation_error = validation_errors[optimal_lr_idx]

        # Print the optimal learning rate and corresponding validation error
        print(f'Optimal Learning Rate: {optimal_lr}')
        print(f'Cross-validated Training Error with Optimal LR: {train_errors[optimal_lr_idx]}')
        print(f'Cross-validated Validation Error with Optimal LR: {optimal_validation_error}')