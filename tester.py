__author__ = 'monicameyer'


def transform_sklearn_dictionary(input_dict):
    X = input_dict['data']
    y = input_dict['target']
    return X, y


def transform_csv(data, target_col='target', ignore_cols=None):
    # Create list of feature names by excluding target name and ignored columns
    X_names = list(data.columns.values)
    X_names.remove(target_col)
    if ignore_cols is not None:
        for name in ignore_cols:
            X_names.remove(name)

    # Subset data to only contain feature columns of interest
    X = data[X_names]

    # Initialize dictionary
    my_dictionary = {}

    # Transform pandas dataframe into list of lists
    my_dictionary['data'] =[]
    for i in xrange(len(X.index)):
        my_dictionary['data'].append(list(X.iloc[i]))

    # Target and target names as lists
    my_dictionary['target'] = list(data[target_col])
    my_dictionary['target_names'] = list(set(data[target_col]))

    # Generic description
    my_dictionary['DESCR'] = "This dataset is reformatted to have specific keys used by sklearn."

    return my_dictionary
