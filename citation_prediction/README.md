# OpenAlex Data Processing and Loading Toolkit

A comprehensive toolkit for downloading, processing, and loading academic paper data from the [OpenAlex API](https://openalex.org/). This repository provides two main components:

1. **`openalex.py`**: A script to fetch and process data from the OpenAlex API based on specified journals and publication years.
2. **`dataloader.py`**: A PyTorch-compatible data loader to handle the processed JSON data, facilitating seamless integration into machine learning workflows.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Download and Processing (`openalex.py`)](#1-data-download-and-processing-openalexpy)
  - [2. Data Loading (`dataloader.py`)](#2-data-loading-dataloaderpy)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Know Limations](#known-limitations)

---

## Features

### `openalex.py`

- **Fetch Data from OpenAlex API**: Retrieve academic papers based on specified journals and publication years.
- **Data Filtering**:
  - Filter papers by type (e.g., articles) and presence of abstracts.
  - Validate page ranges to exclude short papers.
- **Abstract Reconstruction**: Reconstruct abstracts from inverted indexes.
- **Percentile Calculation**: Add percentile fields based on citation counts a specified number of years after publication.
- **Impact Classification**: Classify papers into high and low impact based on citation percentiles.
- **Metadata Generation**: Generate comprehensive metadata summarizing the data processing. One `metadata.json` file will be geneated per experiment.

### `dataloader.py`

- **Custom Dataset**: A PyTorch `Dataset` class tailored for handling JSON data.
- **Data Splitting**: Split data into training, validation, and test sets based on configurable proportions.
- **DataLoaders**: Generate PyTorch `DataLoader` objects for efficient batching and shuffling.
- **Percentile Integration**: Optionally include percentile fields for citation counts at specified years after publication.

---

## Requirements

Ensure you have Python 3.7 or higher installed. The following Python packages are required:

- `requests`
- `scikit-learn`
- `torch`
- `scipy`

You can install these dependencies using `pip`:

```bash
pip install -r requirements.txt
```

*Alternatively, install them individually:*

```bash
pip install requests scikit-learn torch scipy
```

---

## Installation

1. **Clone the Repository**

2. **Install Dependencies**

   As mentioned in the [Requirements](#requirements) section.

3. **Directory Structure**

   Ensure you have a directory structure similar to the following:

   ```
   impact_prediction/
   ├── dataloader.py
   ├── openalex.py
   ├── requirements.txt
   └── README.md
   ```

---

## Usage

### 1. Data Download and Processing (`openalex.py`)

This script fetches data from the OpenAlex API based on specified parameters and processes it for further analysis or machine learning tasks.

#### **Command-Line Arguments**

- `--save_path`: *(str, optional)* Absolute path to save the data. Defaults to the current date in `YYYY-MM-DD` format.
- `--percentage`: *(float, required)* Percentage of top and bottom citations of papers to include.
- `--journal_list`: *(str, required)* List of journals to include. Provide one or more journal names separated by space.
- `--year_start`: *(int, required)* Start year for the data collection.
- `--year_end`: *(int, required)* End year for the data collection.
- `--years_after`: *(int, required)* Number of years after publication to calculate citation impact.

#### **Example Usage**

```bash
python openalex.py \
  --save_path "./data" \
  --percentage 10 \
  --journal_list "Nature" "Science" \
  --year_start 2012 \
  --year_end 2024 \
  --years_after 2
```

*This command will:*

- Save the processed data in the `./data` directory.
- Include the top and bottom 10% of citations.
- Fetch data from the journals "Nature" and "Science".
- Collect data from the years 2015 to 2020.
- Calculate citation impact 3 years after publication.

---

### 2. Data Loading (`dataloader.py`)

The `dataloader.py` script provides a custom PyTorch `DataLoader` to handle the processed JSON data, facilitating seamless integration into machine learning workflows.

#### **Usage**

You can use the `CustomDataLoader` class within your Python scripts to load and prepare data.

#### **Example Usage**

```python
from dataloader import CustomDataLoader

# Specify the directory where JSON data is stored
data_directory = "./data"

# Initialize the CustomDataLoader
data_loader = CustomDataLoader(
    directory=data_directory,
    train_size=0.7,
    valid_size=0.15,
    test_size=0.15,
    batch_size=32,
    random_state=42
)

# Retrieve DataLoader objects
train_loader, valid_loader, test_loader = data_loader.get_loaders()

# Example: Iterate through the training DataLoader
for batch in train_loader:
    # Replace the following line with your training logic
    inputs, labels = batch['abstract'], batch['high_impact']
    # Your training code here
```

#### **Parameters Explained**

- `directory`: Path to the directory containing processed JSON files.
- `train_size`: Proportion of data to use for training (default: 0.7).
- `valid_size`: Proportion of data to use for validation (default: 0.15).
- `test_size`: Proportion of data to use for testing (default: 0.15).
- `batch_size`: Number of samples per batch (default: 32).
- `random_state`: Seed for reproducibility (default: 42).

---

## Examples

### 1. Downloading and Processing Data

```bash
python openalex.py \
  --save_path "./processed_data" \
  --percentage 5 \
  --journal_list "IEEE Transactions on Neural Networks" "Journal of Machine Learning Research" \
  --year_start 2018 \
  --year_end 2022 \
  --years_after 2
```

### 2. Loading Data for Training

```python
from dataloader import CustomDataLoader

# Initialize data loader
data_loader = CustomDataLoader(
    directory="./processed_data",
    train_size=0.8,
    valid_size=0.1,
    test_size=0.1,
    batch_size=64,
    random_state=123
)

# Get DataLoader objects
train_loader, valid_loader, test_loader = data_loader.get_loaders()

# Training loop example
for epoch in range(num_epochs):
    for batch in train_loader:
        # Your training code here
        pass
```

---

## Project Structure

```
openalex-data-toolkit/
├── dataloader.py       # Custom PyTorch DataLoader for JSON data
├── openalex.py         # Script to download and process data from OpenAlex API
├── requirements.txt    # Python dependencies
└── README.md           # This README file
```

---

## Known Limitations
- The OpenAlex API may fetch wrong abtracts. For example, when downloading data from the journal Nature Climate Change, some of the the fetched abstracts are different than the paper's actual abstracts. Therefore, please check the data manually after downloading before use. 
   - In the json outputs, you can click on the "id" field in each article data point, which is a URL to the OpenAlex site. This page contains some fetched information of this article.
   - From this site, you can further click on the 'HTML' or 'PDF' buttons to check the original article, and compare if any fields differ from the fetched data.

---