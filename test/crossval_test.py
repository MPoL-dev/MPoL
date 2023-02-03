from mpol.crossval import CrossValidate


def test_crossvalclass_split(coords, gridder, dataset, generic_parameters):
    # using the CrossValidate class, split a dataset into train/test subsets

    crossval_pars = generic_parameters["crossval_pars"]

    cross_validator = CrossValidate(coords, gridder, **crossval_pars)
    test_train_datasets = cross_validator.split_dataset(dataset)


def test_crossvalclass_kfold(coords, gridder, dataset, generic_parameters):
    # using the CrossValidate class, perform k-fold cross-validation

    crossval_pars = generic_parameters["crossval_pars"]
    # reset some keys to bypass functionality tested elsewhere and speed up test
    crossval_pars["lambda_guess"] = None
    crossval_pars["epochs"] = 11

    cross_validator = CrossValidate(coords, gridder, **crossval_pars)
    test_train_datasets = cross_validator.split_dataset(dataset)
    cv_score, all_scores, loss_histories = cross_validator.run_crossval(test_train_datasets)