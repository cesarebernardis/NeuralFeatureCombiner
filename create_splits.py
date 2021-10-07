import numpy as np
import os

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

save_dataset = True
test_split = True
hybrid_split = True
validation_split = True
collaborative_split = True


def print_stats(dataset):
    urm = dataset.get_URM()
    icm = dataset.get_ICM()

    urm.eliminate_zeros()

    print("------------------------")
    print(dataset.get_name())
    print("Users:", urm.shape[0])
    print("Items:", urm.shape[1])
    print("Features:", icm.shape[1])
    print("Interactions:", len(urm.data))
    print("Density:", len(urm.data) / (urm.shape[0] * urm.shape[1]))
    print("------------------------")


if __name__ == "__main__":

    for dataset_config in EXPERIMENTAL_CONFIG['datasets']:

        datareader = dataset_config['datareader']()
        postprocessings = dataset_config['postprocessings']
        subsampling_pp = dataset_config['subsampling_postprocessings']

        dataset = datareader.load_data(postprocessings=postprocessings)
        if save_dataset:
            dataset.save_data()

        print_stats(dataset)

        for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
            random_seed = fold+1
            if hybrid_split:
                for splitter in EXPERIMENTAL_CONFIG['hybrid_split_perc']:
                    train, test = splitter.split(dataset, random_seed=random_seed + 100)
                    splitter.save_split([train, test], filename_suffix="_{}".format(fold))
            if test_split:
                for splitter in EXPERIMENTAL_CONFIG['cold_split_perc']:
                    train, test = splitter.split(dataset, random_seed=random_seed)
                    splitter.save_split([train, test], filename_suffix="_{}".format(fold))

        splitter = EXPERIMENTAL_CONFIG['cold_split']
        train, test = splitter.load_split(datareader, postprocessings=postprocessings, filename_suffix="_{}".format(0))

        basepath = splitter.get_complete_default_save_folder_path(datareader, postprocessings=postprocessings)

        if validation_split:
            v_train, v_test = splitter.split(train, random_seed=42)
            splitter.save_split([v_train, v_test], save_folder_path=basepath + splitter.get_name() + os.sep)
            if subsampling_pp is not None:
                sub_basepath = splitter.get_complete_default_save_folder_path(datareader, postprocessings=postprocessings)
                subtrain = train
                for pp in subsampling_pp:
                    subtrain = pp.apply(subtrain)
                    sub_basepath += pp.get_name() + os.sep
                sub_basepath += splitter.get_name() + os.sep
                v_train, v_test = splitter.split(subtrain, random_seed=88)
                splitter.save_split([v_train, v_test], save_folder_path=sub_basepath)

        v_train, v_test = splitter.load_split(datareader, save_folder_path=basepath, postprocessings=postprocessings)

        collaborative_splitter = EXPERIMENTAL_CONFIG['warm_split']
        coll_basepath = basepath + splitter.get_name() + os.sep + collaborative_splitter.get_name() + os.sep

        if collaborative_split:
            c_train, c_test = collaborative_splitter.split(v_train, random_seed=42)
            collaborative_splitter.save_split([c_train, c_test], save_folder_path=coll_basepath)

    for dataset_config in EXPERIMENTAL_CONFIG['reduced_datasets']:

        datareader = dataset_config['datareader']()
        postprocessings = dataset_config['postprocessings']

        dataset = datareader.load_data(postprocessings=postprocessings)
        if save_dataset:
            dataset.save_data()

        print_stats(dataset)

