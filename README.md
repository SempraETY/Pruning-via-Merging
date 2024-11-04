# Pruning-via-Merging-Compressing-LLMs-via-Manifold-Alignment-Based-Layer-Merging

## Overview

This project provides a comprehensive pipeline for **fusing layers in pre-trained causal language models** using manifold learning techniques. Layer fusion aims to reduce the number of layers in a model by intelligently combining similar layers, potentially leading to more efficient models without significant loss in performance.

The pipeline involves:

1. **Model Evaluation:** Assessing the model's performance on multiple-choice datasets.
2. **Activation Extraction:** Capturing activations from each model layer for analysis.
3. **Manifold Learning:** Applying diffusion kernels to embed high-dimensional activations into lower-dimensional spaces.
4. **Similarity Computation:** Calculating similarities between layers based on mutual information and entropy estimates.
5. **Layer Fusion:** Combining layers with high similarity to create a more compact model.
6. **Saving the Fused Model:** Exporting the modified model for future use.

## Table of Contents

- [Pruning-via-Merging-Compressing-LLMs-via-Manifold-Alignment-Based-Layer-Merging](#pruning-via-merging-compressing-llms-via-manifold-alignment-based-layer-merging)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Data Format](#data-format)
    - [Directory Structure:](#directory-structure)
    - [CSV File Format:](#csv-file-format)
  - [Usage](#usage)
    - [Command-Line Arguments:](#command-line-arguments)
    - [Example Command:](#example-command)
  - [Process Workflow](#process-workflow)
  - [Outputs](#outputs)

## Features

- **Model Compatibility:** Supports causal language models from Hugging Face's Transformers library.
- **Customizable Fusion:** Specify the number of layers to fuse based on similarity metrics.
- **Efficient Computation:** Utilizes adaptive chunking for memory-efficient distance calculations.
- **Robustness:** Includes mechanisms to handle numerical stability issues during matrix computations.
- **Comprehensive Logging:** Records detailed information about the fusion process for traceability.

## Requirements

- **Python:** 3.8 or higher
- **Libraries:**
  - `torch`
  - `transformers`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tqdm`
  - `pickle`
  - `logging`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/layer-fusion](https://github.com/SempraETY/Pruning-via-Merging-Compressing-LLMs-via-Manifold-Alignment-Based-Layer-Merging.git
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, you can install the necessary libraries using:*

   ```bash
   pip install torch transformers numpy pandas scikit-learn tqdm
   ```

## Data Format

The pipeline expects data organized in a specific directory structure with CSV files for different subjects. Each subject should have corresponding development (`dev`) and test (`test`) CSV files.

### Directory Structure:

```
data/
├── dev/
│   ├── subject1_dev.csv
│   ├── subject2_dev.csv
│   └── ...
└── test/
    ├── subject1_test.csv
    ├── subject2_test.csv
    └── ...
```

### CSV File Format:

Each CSV file should follow this structure:

- **Column 0:** Question prompt (string)
- **Columns 1-4:** Multiple-choice options labeled A to D
- **Column 5:** Correct answer label (e.g., "A", "B", "C", or "D")

*Example Row:*

| Column 0                | Column 1 | Column 2 | Column 3 | Column 4 | Column 5 |
|-------------------------|----------|----------|----------|----------|----------|
| "What is the capital..."| "A. ..." | "B. ..." | "C. ..." | "D. ..." | "B"      |

## Usage

Run the main script with appropriate command-line arguments to perform layer fusion.

### Command-Line Arguments:

- `--ntrain`, `-k`: **(int, default=5)** Number of training examples to include in prompts.
- `--ngpu`, `-g`: **(int, default=4)** Number of GPUs to use.
- `--model_path`: **(str, default="/data/yangzhao/point/baichuan/Meta-Llama-3-70B")** Path to the pre-trained model.
- `--num_tasks`, `-n`: **(int, default=57)** Number of MMLU tasks to process.
- `--num_samples`, `-m`: **(int, default=1)** Number of samples per task.
- `--data_dir`, `-d`: **(str, default="data")** Directory containing the data.
- `--num_layer`, `-i`: **(int, default=1)** Number of layers to fuse.

### Example Command:

```bash
python pipeline.py --ntrain 10 --ngpu 2 --model_path "/path/to/model" --num_tasks 50 --num_samples 5 --data_dir "./data" --num_layer 2
```

## Process Workflow

1. **Initialization:**
   - Parses command-line arguments.
   - Sets up directories for embeddings, fusion information, and merged weights.
   - Configures logging and sets random seeds for reproducibility.

2. **Model and Tokenizer Loading:**
   - Loads the tokenizer and pre-trained causal language model from the specified `model_path`.

3. **Data Processing:**
   - Iterates through each subject's test data.
   - For each selected sample, formats the prompt and extracts activations from all model layers.

4. **Manifold Learning:**
   - Applies a diffusion kernel to the stacked activations of each layer to obtain lower-dimensional embeddings.

5. **Similarity Computation:**
   - Calculates a similarity matrix based on normalized pointwise information bottleneck (NPIB) between layers.

6. **Layer Fusion:**
   - Identifies pairs of layers with high similarity.
   - Fuses specified layers by blending their weights based on computed fusion ratios.
   - Updates the model configuration and removes fused layers.

7. **Saving the Fused Model:**
   - Saves the modified model's state dictionary and configuration to the designated directory.

## Outputs

- **Embeddings Directory:** Contains pickled files of embedded activations for each layer.
- **Fusion Info Directory:** Stores logs detailing the fusion process.
- **Merged Weights Directory:** Holds the fused model's configuration and the `pytorch_model.bin` file.

**Example Structure:**

```
EMNLP2024/
└── layer_fus/
    └── al/
        └── Meta-Llama-3-70B/
            └── fused_2_layers/
                └── iteration/
                    ├── embeddings/
                    │   ├── layer_0_embedded.pkl
                    │   ├── layer_1_embedded.pkl
                    │   └── ...
                    ├── fusion_info/
                    │   └── experiment.log
                    └── merged_weights/
                        ├── config.json
                        └── pytorch_model.bin


                        
```
