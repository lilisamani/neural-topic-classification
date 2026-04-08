import argparse
import os


def load_input_files(directory="data"):
    source_address_prefix = "https://huggingface.co/datasets/Davlan/sib200/raw/main/data/zho_Hans/"

    filenames = {"dev": "dev.tsv", "test": "test.tsv", "train": "train.tsv"}

    # if the "data" folder does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # download the files if they do not exist
    output = {}
    for key, file in filenames.items():
        file_path = os.path.join(directory, file)
        if not os.path.exists(file_path):
            os.system(f"wget -nc {source_address_prefix}{file} -O {file_path}")
        output[key] = file_path

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural topic classification data downloader")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory to store/load data files"
    )
    args = parser.parse_args()

    input_files = load_input_files(args.data_dir)
    print(f"Data loaded successfully into directory: {args.data_dir}")
