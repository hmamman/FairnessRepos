import os
import sys

os.chdir('.')
sys.path.append(os.getcwd())

from Phemus import *


def main():
    num_param = 8
    sensitive_param_idx = 5  # Starts at O
    sensitive_param_name = "Gender"
    col_to_be_predicted = "LeaveOrNot"
    dataset_dir = "Employee.csv"
    model_type = None
    # num_params, sensitive_param_idx, model_type, sensitive_param_name, col_to_be_predicted, dataset_dir, sensitive_param_idx_list = [], sensitive_param_name_list = []

    dataset = Dataset(num_param, sensitive_param_idx, model_type, sensitive_param_name, col_to_be_predicted,
                      dataset_dir, sensitive_param_idx_list=[], sensitive_param_name_list=[])

    pkl_dir = 'Employee_DecisionTree_Original.pkl'
    improved_pkl_dir = 'Employee_DecisionTree_Original_Improved.pkl'

    threshold = 0
    perturbation_unit = 1
    global_iteration_limit = 1000  # needs to be at least 1000 to be effective
    local_iteration_limit = 100

    num_trials = 100
    samples = 100

    # run_aequitas_fully_direct(dataset, perturbation_unit, pkl_dir, improved_pkl_dir, threshold, \
    #                           global_iteration_limit, local_iteration_limit, num_trials, samples)

    num_params = 8
    sensitive_param_idx = 5  # Starts at O
    sensitive_param_name = "Gender"
    col_to_be_predicted = "LeaveOrNot"
    dataset_dir = "Employee.csv"
    model_type = "DecisionTree"
    os.chdir('Examples')
    sys.path.append(os.getcwd())

    dataset = Dataset(num_params=num_params, sensitive_param_idx=sensitive_param_idx, \
                      model_type=model_type, sensitive_param_name=sensitive_param_name, \
                      col_to_be_predicted=col_to_be_predicted, dataset_dir=dataset_dir)

    pkl_dir = 'Employee_DecisionTree_Original.pkl'
    improved_pkl_dir = 'Employee_DecisionTree_Original_Improved.pkl'
    retrain_csv_dir = 'Employee_Retraining_Dataset.csv'
    plot_dir = 'Employee_Fairness_Plot.png'

    perturbation_unit = 1

    num_trials = 1000
    samples = 100
    global_iteration_limit = 1000  # needs to be at least 1000 to be effective
    local_iteration_limit = 100
    threshold = 0

    retrain_csv_dir = 'Employee_Retraining_Dataset.csv'
    run_aequitas(dataset=dataset, perturbation_unit=perturbation_unit,
                 pkl_dir=pkl_dir, improved_pkl_dir=improved_pkl_dir,
                 retrain_csv_dir=retrain_csv_dir, plot_dir=plot_dir, mode="Fully",
                 threshold=threshold,
                 global_iteration_limit=global_iteration_limit,
                 local_iteration_limit=local_iteration_limit,
                 num_trials=num_trials,
                 samples=samples
                 )


if __name__ == "__main__":
    main()