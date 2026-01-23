# Author: Mehtab "Shahan" Iqbal
# Affiliation: School of Computing, Clemson University
# Created on: 20th January 2026
# This content of this file is used by the run_experiments.py script to
# synthesize outlier records. To run this wihout the python driver program,
# the necessary parameters must be provided.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Missing arguments: input_path output_path [seed]")
}

input_path <- args[1]
output_path <- args[2]
random_seed <- if (length(args) > 2) as.integer(args[3]) else 2026

suppressPackageStartupMessages(library(synthpop))
suppressPackageStartupMessages(library(arrow))

# 1. Load Data
# We use feather instead of CSV to leverage the read/write speed, and transfer
# reliability (Python -> R).
data <- read_feather(input_path)
char_cols <- sapply(data, is.character)
if (any(char_cols)) {
    data[char_cols] <- lapply(data[char_cols], as.factor)
}

# Convert integer-coded categories
categorical_vars <- c(
  "DISABL1", "CLASS", "INDUSTRY_CLASS", "OCCUP_CLASS",
  "RLABOR", "RELAT1", "HOUR89_CAT", "WEEK89_CAT"
)
for (col in categorical_vars) {
  if (col %in% names(data)) {
    data[[col]] <- as.factor(data[[col]])
  }
}
numeric_vars <- c("YEARSCH", "AGE")
target_var <- "INCOME1"

# 2. Isolate ID and Exclude AGE (as requested)
# We assume 'ORIGINAL_ID' is present to preserve order
ids <- data$ORIGINAL_ID

# We drop ORIGINAL_ID, HOUR89, and WEEK89..
vars_to_exclude <- c("ORIGINAL_ID", "HOUR89", "WEEK89")
data_to_syn <- data[, !(names(data) %in% vars_to_exclude)]

# 3. Define Visit Sequence
# synthpop's default CART is usually smart enough, but we enforce the dependent
# variable last.
visit_seq <- c(categorical_vars, numeric_vars, target_var)

# k = nrow ensures 1-t-1 record generation
k <- nrow(data_to_syn)
# 4. Run a "Dry Run" (m=0) to get the default matrix structure
# This creates the correct matrix dimensions and row/col names automatically
ini <- syn(data_to_syn,
  k = k,
  method = "cart",
  visit.sequence = visit_seq,
  m = 0
)

# 4. Extract and Customize the Predictor Matrix
pred_mat <- ini$predictor.matrix

# We want to be sure no accidental circular relationships exist:
# Reset the row for INCOME1 to 0
pred_mat["INCOME1", ] <- 0

# Set 1s for your specific predictors
predictors <- c(categorical_vars, numeric_vars)
pred_mat["INCOME1", predictors] <- 1

# Verify the matrix (Optional)
print("Predictor Matrix for INCOME1:")
print(pred_mat["INCOME1", ])

# 4. Run the synthesis
# Note: We must pass the SAME visit.sequence we used to generate the matrix
syn_obj <- syn(data_to_syn,
  method = "cart",
  k = nrow(data_to_syn),
  predictor.matrix = pred_mat,
  visit.sequence = visit_seq,
  proper = TRUE,
  seed = random_seed
)

# 5. Extract and Re-attach ID
syn_data <- syn_obj$syn
syn_data$ORIGINAL_ID <- ids

# 6. Export
write.csv(syn_data, output_path, row.names = FALSE)