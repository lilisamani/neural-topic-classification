# neural-topic-classification
LT2222 Assignment 3: Neural Topic Classification for Simplified Chinese

## Below are the step by step instructions to run the scripts for this assignment.
Before you start, make sure you are in the root directory of the repository (the one containing this README file).

### Step 1: Download the input files
Run the following command in the terminal to download the input files (train, dev, test, and labels)
into a directory named "data":
```bash
$ python3 download_input_files.py --data-dir data
```
This will create a "data" directory in your current working directory and download the necessary
files into it.
```
data/dev.tsv                                                           100%[============================================================================================================================================================================>]  12.35K  --.-KB/s    in 0s

2026-04-08 22:34:20 (207 MB/s) - ‘data/dev.tsv’ saved [12647/12647]

--2026-04-08 22:34:20--  https://huggingface.co/datasets/Davlan/sib200/raw/main/data/zho_Hans/test.tsv
Resolving huggingface.co (huggingface.co)... 65.9.46.54, 65.9.46.59, 65.9.46.41, ...
Connecting to huggingface.co (huggingface.co)|65.9.46.54|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 27922 (27K) [text/plain]
Saving to: ‘data/test.tsv’

data/test.tsv                                                          100%[============================================================================================================================================================================>]  27.27K  --.-KB/s    in 0.007s

2026-04-08 22:34:20 (4.09 MB/s) - ‘data/test.tsv’ saved [27922/27922]

--2026-04-08 22:34:20--  https://huggingface.co/datasets/Davlan/sib200/raw/main/data/zho_Hans/train.tsv
Resolving huggingface.co (huggingface.co)... 65.9.46.59, 65.9.46.41, 65.9.46.108, ...
Connecting to huggingface.co (huggingface.co)|65.9.46.59|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 96974 (95K) [text/plain]
Saving to: ‘data/train.tsv’

data/train.tsv                                                         100%[============================================================================================================================================================================>]  94.70K  --.-KB/s    in 0.1s

2026-04-08 22:34:21 (895 KB/s) - ‘data/train.tsv’ saved [96974/96974]

Data loaded successfully into directory: data
```
### Step 2: Run the sentence embedding script
Run the following command in the terminal to compute sentence embeddings for the input files and save them to an output file named "embeddings.tsv":
```bash
$ python3 sentence_embeddings.py --dimensionality 512 --train-file data/train.tsv --dev-file data/dev.tsv --test-file data/test.tsv --output-file embeddings.tsv
```
This will compute sentence embeddings for the train, dev, and test files using the Word2Vec
