from torch.utils.data import DataLoader
from multiprocessing import freeze_support
import os
from datasets import add_compute_stats
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from pathlib import Path
from experiments.transformations.utils.misc import classes_sets, exp_categorization_task
from experiments.transformations.utils.datasets import (
    SubclassImageFolder,
    SameDifferentSampler,
    add_unpack_transf_info,
    get_transforms,
)
import framework_utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
desired_width = 320
np.set_printoptions(linewidth=desired_width)

this_file_path = Path(globals().get("__file__", "./_")).absolute().parent


def get_weighted_sampler(dataset):
    weights_class = 1 / np.array(
        [
            np.sum([True for i in dataset.samples if i[1] == c])
            for c in range(dataset.num_classes)
        ]
    )
    weights_dataset = [weights_class[i[1]] for i in dataset.samples]
    return WeightedRandomSampler(
        weights=weights_dataset, num_samples=len(weights_dataset), replacement=True
    )


def get_name_samples_file_train(name_generator, exp):
    return f"{str(this_file_path)}/samples/{name_generator}_{exp.classes_set}_numO{exp.max_objs_per_class_train}_numV{exp.num_viewpoints_train}_S{exp.seed}.sample"


def get_name_samples_file_test(name_generator, exp):
    return f"{str(this_file_path)}/samples/{name_generator}_{exp.classes_set}_numO{exp.max_objs_per_class_test}_numV{exp.num_viewpoints_test}_S{exp.seed}.sample"


def cat_exp(**experiment_params):
    def get_sampler_or_batch_sampler(dataset):
        if exp.network_name == "samediffnet":
            return dict(
                batch_sampler=SameDifferentSampler(
                    batch_size=exp.batch_size,
                    prob_same=0.5,
                    rebalance_classes=True,
                    subclasses=dataset.subclasses,
                    subclasses_names=dataset.subclasses_names,
                    dataset=dataset,
                )
            )
        else:
            return dict(shuffle=True, batch_size=exp.batch_size)

    exp = exp_categorization_task(**experiment_params)
    name_folder = exp.shapenet_folder

    stats = {
        "mean": [0.06229676, 0.0607271, 0.05646703],
        "std": [0.14454809, 0.14061172, 0.12994126],
    }

    add_PIL_transforms, add_T_transforms = None, None
    if "0" in exp.transformations:  # not currently used in the paper
        tt = "".join(
            [i for i in exp.transformations if i != "t" and i != "r" and i != "s"]
        )
    else:
        tt = exp.transformations
        add_PIL_transforms, add_T_transforms = get_transforms(tt, info=False)

    train_args = dict(
        add_PIL_transforms=add_PIL_transforms,
        add_tensor_transforms=add_T_transforms,
        save_stats_file=None,
        num_objects_per_class=exp.max_objs_per_class_train,
        stats=stats,
        name_classes=classes_sets[exp.classes_set],
        num_viewpoints_per_object=exp.num_viewpoints_train,
        name_generator=f"train {exp.classes_set}",
        num_image_calculate_mean_std=300 if exp.use_cuda else 100,
        root=f"./data/ShapeNet2D/{name_folder}/train",
        save_load_samples_filename=get_name_samples_file_train("train", exp),
    )

    train_dataset = add_compute_stats(SubclassImageFolder)(**train_args)

    training_loader = DataLoader(
        train_dataset,
        **get_sampler_or_batch_sampler(train_dataset),
        num_workers=8 if exp.use_cuda else 0,
        timeout=60 if exp.use_cuda else 0,
        pin_memory=True,
    )

    ################ TESTING ##################
    add_PIL_tr_info_test, add_T_tr_inf_test = get_transforms(
        exp.transformations_test, info=True
    )
    add_PIL_tr_info_train, add_T_tr_inf_train = get_transforms(
        exp.transformations, info=True
    )

    args = dict(
        num_objects_per_class=exp.max_objs_per_class_test,
        stats=stats,
        num_viewpoints_per_object=exp.num_viewpoints_test,
        add_PIL_transforms=add_PIL_tr_info_test,
        add_tensor_transforms=add_T_tr_inf_test,
    )
    dataset_test = []

    # we only run this when we are on the server as I don't have this specific dataset locally
    if exp.use_cuda:
        dataset_test += [
            add_unpack_transf_info(add_compute_stats(SubclassImageFolder))(
                name_generator=f"test same objs diff vp {exp.classes_set}",
                take_specific_azi_incl=False,
                selected_objects=train_dataset.selected_objects,
                root=f"./data/ShapeNet2D/{name_folder}/testSameObjsDiffViewpoints",
                save_load_samples_filename=get_name_samples_file_test(
                    "test_same_objs_diff_vp", exp
                ),
                name_classes=classes_sets[exp.classes_set],
                **args,
            )
        ]

    dataset_test += [
        add_unpack_transf_info(add_compute_stats(SubclassImageFolder))(
            name_generator=f"test diff objs {exp.classes_set}",
            root=f"./data/ShapeNet2D/{name_folder}/testDiffObjects",
            save_load_samples_filename=get_name_samples_file_test(
                "test_diff_objs", exp
            ),
            name_classes=classes_sets[exp.classes_set],
            **args,
        )
    ]

    dataset_test += [
        add_unpack_transf_info(add_compute_stats(SubclassImageFolder))(
            name_generator=f"test diff classes {exp.diff_classes_set}",
            root=f"./data/ShapeNet2D/{name_folder}/train",
            save_load_samples_filename=get_name_samples_file_test(
                f"train {exp.diff_classes_set}", exp
            ),
            name_classes=classes_sets[exp.diff_classes_set],
            **args,
        )
    ]

    train_args.update(
        dict(
            add_PIL_transforms=add_PIL_tr_info_train,
            add_tensor_transforms=add_T_tr_inf_train,
            name_generator=f"test like train {exp.classes_set}",
        )
    )
    dataset_test += [
        add_unpack_transf_info(add_compute_stats(SubclassImageFolder))(**train_args)
    ]

    if exp.transformations != exp.transformations_test:
        train_args.update(
            dict(
                add_PIL_transforms=add_PIL_tr_info_test,
                add_tensor_transforms=add_T_tr_inf_test,
                name_generator=f"transformed trs train {exp.classes_set}",
            )
        )
        dataset_test += [
            add_unpack_transf_info(add_compute_stats(SubclassImageFolder))(**train_args)
        ]

    testing_loader_list = [
        DataLoader(
            dataset,
            **get_sampler_or_batch_sampler(dataset),
            num_workers=8 if exp.use_cuda else 0,
            timeout=60 if exp.use_cuda else 0,
            pin_memory=True,
        )
        for dataset in dataset_test
    ]

    ###########################################
    try:
        for run in range(exp.num_runs):
            framework_utils.weblogging_plot_generators_info(
                training_loader,
                testing_loader_list,
                exp.weblogger,
                num_batches_to_log=np.max((1, round(20 / exp.batch_size))),
            )
            if exp.do_train:
                net = exp.train(
                    training_loader,
                    log_text=training_loader.dataset.name_generator,
                    test_loaders=testing_loader_list,
                )
            else:
                net = exp.get_net(new_num_classes=training_loader.dataset.num_classes)
            exp.test(net, testing_loader_list)
            if run < (exp.num_runs - 1):
                exp.new_run()
        exp.save_all_runs_and_stop()
        exp.weblogger["STATUS"] = "Success"
        return exp
    except:
        exp.weblogger["STATUS"] = "Failed"
        raise


if __name__ == "__main__":
    freeze_support()
    cat_exp()
