from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import pandas as pd

from aif360.datasets import StandardDataset



'''
This mappings should be modified

'''
default_mappings = {
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}], 
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
                                 {1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'City', 0.0: 'Country'}]
}

class MexicoDataset(StandardDataset):
    """Mexico Poverty Dataset.

    See :file:`aif360/data/raw/adult/README.md`. # To be editted
    """

    def __init__(self, 
                label_name='Outcome',
                 favorable_classes=[1, 1, 1],
                 protected_attribute_names=['ftype', 'urban', 'young'],
                 privileged_classes=[[1], [1], [1]],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[''], custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        Examples:
            The following will instantiate a dataset which uses the `fnlwgt`
            feature:

            >>> from aif360.datasets import AdultDataset
            >>> ad = AdultDataset(instance_weights_name='fnlwgt',
            ... features_to_drop=[])
            WARNING:root:Missing Data: 3620 rows removed from dataset.
            >>> not np.all(ad.instance_weights == 1.)
            True

            To instantiate a dataset which utilizes only numerical features and
            a single protected attribute, run:

            >>> single_protected = ['sex']
            >>> single_privileged = [['Male']]
            >>> ad = AdultDataset(protected_attribute_names=single_protected,
            ... privileged_classes=single_privileged,
            ... categorical_features=[],
            ... features_to_keep=['age', 'education-num'])
            >>> print(ad.feature_names)
            ['education-num', 'age', 'sex']
            >>> print(ad.label_names)
            ['income-per-year']

            Note: the `protected_attribute_names` and `label_name` are kept even
            if they are not explicitly given in `features_to_keep`.

            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: '>50K', 0.0: '<=50K'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> ad = AdultDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'data', 'raw', 'mexico', 'mexico_train.csv')
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'data', 'raw', 'mexico', 'mexico_test.csv')


        try:
            train = pd.read_csv(train_path, na_values=na_values)
            test = pd.read_csv(test_path, na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'adult'))))
            import sys
            sys.exit(1)
        df = pd.concat([train, test], ignore_index=True)

        super(MexicoDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
