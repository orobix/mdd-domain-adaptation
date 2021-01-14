import argparse

from mdd.utils.download import datasets, download_with_resume

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="office-31", choices=["office-31", "image-clef"]
    )
    args = parser.parse_args()
    if args.dataset not in datasets.keys():
        raise ValueError(
            "The '--dataset' argument must be one of: "
            + " ".join(datasets.keys())
        )
    download_with_resume(
        datasets[args.dataset]["url"],
        datasets[args.dataset]["dest"],
        args.dataset,
    )
