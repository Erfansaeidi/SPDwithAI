# SPDwithAI

This repository contains datasets and resources for training Software Defect Prediction (SPD) models using Artificial Intelligence (AI). The datasets included are sourced from the NASA Promise dataset repository and are provided in both CSV and ARFF formats.

## Datasets

The repository includes the following datasets:

- KC1
- JM1
- CM1
- KC2
- PC1

These datasets are crucial for training and evaluating software defect prediction models.

Link to the data repository: [NASA Promise Dataset Repository](http://promise.site.uottawa.ca/SERepository/datasets-page.html)

## Project Structure

The project is organized as follows:

```
SPDwithAI/
│
├── data/
│   ├── KC1.csv
│   ├── JM1.csv
│   ├── CM1.csv
│   ├── KC2.csv
│   └── PC1.csv
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   └── evaluation.py
│
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Jupyter Notebook

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Erfansaeidi/SPDwithAI.git
   cd SPDwithAI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Exploration**:
   Open and run the `data_exploration.ipynb` notebook in the `notebooks/` directory to explore the datasets.

2. **Model Training**:
   Use the `model_training.ipynb` notebook to train and evaluate software defect prediction models.

3. **Scripts**:
   - `data_preprocessing.py`: Contains functions for data preprocessing.
   - `model.py`: Defines the machine learning models.
   - `evaluation.py`: Provides evaluation metrics and functions.

### Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

- [NASA Promise Dataset Repository](http://promise.site.uottawa.ca/SERepository/datasets-page.html) for providing the datasets.
