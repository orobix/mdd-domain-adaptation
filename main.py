import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import mdd.transforms as t
from mdd.data.dataset import CyclicDataset
from mdd.data.domain_adaptation import ImageClef, Office31, OfficeHome
from mdd.models import MDDLitModel


def train(config):
    # Pre-process
    prep_dict = {}
    prep_dict["source"] = t.train(**config["prep"])
    prep_dict["target"] = t.train(**config["prep"])
    prep_dict["test"] = (
        t.test_10crop(**config["prep"])
        if config["data"]["test_10crop"]
        else t.test(**config["prep"])
    )

    # Data
    if config["data"]["dset"] == "image-clef":
        train_source_data = ImageClef(
            domain=config["data"]["source_domain"],
            transform=prep_dict["source"],
            test_10crop=False,
        )
        train_target_data = ImageClef(
            domain=config["data"]["target_domain"],
            transform=prep_dict["target"],
            test_10crop=False,
        )
        test_target_data = ImageClef(
            domain=config["data"]["target_domain"],
            transform=prep_dict["test"],
            test_10crop=config["data"]["test_10crop"],
        )
    elif config["data"]["dset"] == "office-31":
        train_source_data = Office31(
            domain=config["data"]["source_domain"],
            transform=prep_dict["source"],
            test_10crop=False,
        )
        train_target_data = Office31(
            domain=config["data"]["target_domain"],
            transform=prep_dict["target"],
            test_10crop=False,
        )
        test_target_data = Office31(
            domain=config["data"]["target_domain"],
            transform=prep_dict["test"],
            test_10crop=config["data"]["test_10crop"],
        )
    elif config["data"]["dset"] == "office-home":
        train_source_data = OfficeHome(
            domain=config["data"]["source_domain"],
            transform=prep_dict["source"],
            test_10crop=False,
        )
        train_target_data = OfficeHome(
            domain=config["data"]["target_domain"],
            transform=prep_dict["target"],
            test_10crop=False,
        )
        test_target_data = OfficeHome(
            domain=config["data"]["target_domain"],
            transform=prep_dict["test"],
            test_10crop=config["data"]["test_10crop"],
        )

    max_data_length = max(len(train_source_data), len(train_target_data))
    if (
        config["trainer"]["max_steps"] * config["data"]["train_batch_size"]
        < max_data_length
    ):
        print(
            "The number of sampled images ("
            + str(config["trainer"]["max_steps"] * config["data"]["train_batch_size"])
            + ") is less than the available data ("
            + str(max_data_length)
            + ")"
        )
        exit()

    dataset = CyclicDataset(
        train_source_data,
        train_target_data,
        num_iterations=config["trainer"]["max_steps"]
        * config["data"]["train_batch_size"],
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["train_batch_size"],
        num_workers=config["data"]["num_workers"],
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_target_data,
        batch_size=config["data"]["test_batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=config["data"]["num_workers"],
    )

    # MDD model
    model = MDDLitModel(**config["model"], test_10crop=config["data"]["test_10crop"])

    # Model checkpoint every n steps
    checkpoint = ModelCheckpoint(
        monitor="val_acc_epoch",
        save_top_k=1,
        mode="max",
        filename=config["data"]["dset"] + "-{step:d}-{val_acc_epoch:.3f}",
    )
    callbacks_pl = [checkpoint]

    # Trainer
    trainer = Trainer(**config["trainer"], callbacks=callbacks_pl)
    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=test_dataloader,
    )
    trainer.test(model, test_dataloaders=test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    MDDLitModel.add_model_specific_args(parser)
    data_args = parser.add_argument_group("data arguments")
    data_args.add_argument(
        "--dset",
        type=str,
        default="office-31",
        choices=["office-31", "image-clef", "office-home"],
        help="The dataset or source dataset type",
    )
    data_args.add_argument(
        "--source_domain",
        type=str,
        default="amazon",
        help="The source domain",
    )
    data_args.add_argument(
        "--target_domain",
        type=str,
        default="webcam",
        help="The target domain",
    )
    data_args.add_argument(
        "--num_workers",
        type=int,
        default=5,
        help="Pytorch DataLoader num workers",
    )
    data_args.add_argument(
        "--train_batch_size",
        type=int,
        default=36,
        help="Training batch size",
    )
    data_args.add_argument(
        "--test_batch_size",
        type=int,
        default=4,
        help="Testing batch size",
    )
    data_args.add_argument(
        "--test_10crop",
        action="store_true",
        default=False,
        help="Testing with random 10 crop",
    )
    # misc_args = parser.add_argument_group("misc arguments")
    # misc_args.add_argument(
    #     "--snapshot_interval",
    #     type=int,
    #     default=50,
    #     help="save checkpoint every '--snapshost_interval' steps",
    # )

    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {
            a.dest: getattr(args, a.dest, None)
            for a in group._group_actions
            if a.dest != "help"
        }
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    config = {}
    config["trainer"] = vars(arg_groups["optional arguments"])
    config["model"] = vars(arg_groups["model arguments"])
    config["data"] = vars(arg_groups["data arguments"])
    # config["misc"] = vars(arg_groups["misc arguments"])
    config["prep"] = {"resize_size": 256, "crop_size": 224}

    if config["data"]["dset"] == "office-31":
        if (
            ("amazon" in args.source_domain and "webcam" in args.target_domain)
            or ("webcam" in args.source_domain and "dslr" in args.target_domain)
            or ("webcam" in args.source_domain and "amazon" in args.target_domain)
            or ("dslr" in args.source_domain and "amazon" in args.target_domain)
        ):
            config["model"]["scheduler_lr"] = 0.001  # optimal parameters 0.001 default
        elif ("amazon" in args.source_domain and "dslr" in args.target_domain) or (
            "dslr" in args.source_domain and "webcam" in args.target_domain
        ):
            config["model"]["scheduler_lr"] = 0.0003  # optimal parameters 0.0003 default
        config["model"]["num_class"] = 31
    elif config["data"]["dset"] == "image-clef":
        config["model"]["scheduler_lr"] = 0.001  # optimal parameters
        config["model"]["num_class"] = 12
    elif config["data"]["dset"] == "visda":
        config["model"]["scheduler_lr"] = 0.001  # optimal parameters
        config["model"]["num_class"] = 12
        config["model"]["loss_trade_off"] = 1.0
    elif config["data"]["dset"] == "office-home":
        config["model"]["scheduler_lr"] = 0.001  # optimal parameters
        config["model"]["num_class"] = 65
    else:
        raise ValueError(
            "Dataset cannot be recognized. Please define your own dataset here."
        )
    train(config)
