from mpol.crossval import CrossValidate


def test_crossvalclass_split(coords, gridder, dataset, generic_parameters):
    # using the CrossValidate class, split a dataset into train/test subsets

    crossval_pars = generic_parameters["crossval_pars"]

    cross_validator = CrossValidate(coords, gridder, **crossval_pars)
    test_train_datasets = cross_validator.split_dataset(dataset)


