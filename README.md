# Subpop-miner

This repository contains the source code to mining algorithm that identifies
subpopulations where outliers are defined differently than in the rest of the
population and might need adjustments in their protection.

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

1. Wizard step 1: Loading the dataset
![Wizard step 1a](https://raw.githubusercontent.com/shahanM/subpop-miner/main/imgs/wiz1_nofile.png)

**Load data**: The user can load the data from a CSV file. The CSV file must
   contain a header row.

2. Wizard step 2: Selecting the attributes to be used in the analysis
![Wizard step 2](https://raw.githubusercontent.com/shahanM/subpop-miner/main/imgs/wiz2.png)

**Select relevant columns**: The user can select the columns that are
   relevant to the data protection project.

3. Wizard step 3: Indicating the attribute types and the variable subject extreme value protection
![Wizard step 3](https://raw.githubusercontent.com/shahanM/subpop-miner/main/imgs/wiz3.png)

**Indicate data types**: The user needs to indicate the data types of the
   selected columns. The data types are considered in two levels: the first
   level is the general data type (`numeric`, `categorical`), and the second
   level indicates if the variable is `dependent` or `independent`.
   The `dependent` variable is a numerical variable outlier of which must be protected. The `independent` variables are categorical and continuous variables that define subpopulations.

   The variable subject to protection is indicated by the user by selecting the 
   target radio button.

4. Wizard step 4: Setting the parameters for the mining algorithm
![Wizard step 4](https://raw.githubusercontent.com/shahanM/subpop-miner/main/imgs/wiz4.png)