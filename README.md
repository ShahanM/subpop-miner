# Subpop-miner

This repository contains the source code to mining algorithm that identifies
subpopulation where outliers are defined differently that the rest of the
population.

_Note: The user interface is in development and may contain bugs, incomplete
form, and other interface affordances._

## How to setup your environment?

### Requirements

- Python 3.9
- Qt 6.1.2

### Setup

Use the `requirements.txt` file with `pip` to install the necessary packages.

```pip install -r requirements.txt```

## How to run the application?

### Run the application

```python main.py```

### The user interface

The user interface provides 5 step wizard to guide the user through the process.

1. **Load data**: The user can load the data from a CSV file. The CSV file must
   contain a header row.
2. **Select relevant columns**: The user can select the columns that are
   relevant to the data protection project.
3. **Indicate data types**: The user can indicate the data types of the
   selected columns. The data types are considered in two levels: the first
   level is the general data type (`numeric`, `categorical`), and the second
   level indicates if the variable is `dependent` or `independent`.
4. **Select parameters**: The user can select the parameters for the
   subpopulation mining algorithm.
5. **Run the algorithm**: The user can run the algorithm and receive a list of
   subpopulations that that are sensitive to data disclosure.
