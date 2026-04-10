import argparse
from pathlib import Path

import pandas as pd
from gensim.models import FastText


def read_data(file_path: Path):
    if not file_path:
        return pd.DataFrame()
    if file_path.suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


# function that uses gensim to compute sentence embeddings for given input files and saves the
# embeddings to the output file
def compute_sentence_embeddings(
    dimensionality: int,
    train_file: Path,
    dev_file: Path,
    test_file: Path,
):
    train_dt, dev_dt, test_dt = [read_data(file) for file in [train_file, dev_file, test_file]]
    data = []
    for dataset, dataset_name in zip([train_dt, dev_dt, test_dt], ["train", "dev", "test"]):
        for entry in dataset.itertuples():
            data.append([dataset_name, entry.index_id, entry.category, list(entry.text)])
    model = FastText(
        [row[3] for row in data],
        vector_size=dimensionality,
        min_count=1,
        window=13,
        workers=16,
        epochs=2000,
    )
    for index in range(len(data)):
        embedding = model.wv[data[index][3]].mean(axis=0)
        data[index].append(embedding)
    return data


# save embeddings to the output file in pickle format using pandas
def save_embeddings_to_file(data, output_file):
    df = pd.DataFrame(data, columns=["dataset", "index_id", "label", "text", "embedding"])
    df.to_pickle(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentence embeddings script for neural topic classification."
    )
    parser.add_argument(
        "dimensionality",
        type=int,
        help="Embedding dimensionality.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output file path.",
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="Training data file path.",
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=None,
        help="Development data file path.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Test data file path.",
    )

    args = parser.parse_args()

    data = compute_sentence_embeddings(
        args.dimensionality,
        args.train_file,
        args.dev_file,
        args.test_file,
    )

    save_embeddings_to_file(data, args.output_file)
    print(f"Embeddings saved successfully to {args.output_file}")
