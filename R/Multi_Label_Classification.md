# Multi-Label Classification in R: A Step-by-Step Guide  

## Introduction  
Multi-label classification is a type of machine learning problem where each instance (data point) can be associated with multiple labels. Unlike multi-class classification, where each instance is assigned only one label, multi-label classification allows multiple labels per instance.  

In this tutorial, we will:  
1. Load and preprocess the data  
2. Train a multi-label classification model using R  
3. Evaluate the model’s performance  

We will use the `mlr3` and `mlr3measures` packages for handling multi-label classification.

---

## Step 1: Install and Load Required Packages  

```r
install.packages("mlr3")
install.packages("mlr3measures")
install.packages("mlr3learners")
install.packages("data.table")

library(mlr3)
library(mlr3measures)
library(mlr3learners)
library(data.table)
```

---

## Step 2: Load the Dataset  

For this tutorial, we will use the `yeast` dataset from the `mlr3data` package, which is commonly used for multi-label classification tasks. It contains gene expression data with multiple functional classes as labels.  

```r
install.packages("mlr3data")
library(mlr3data)

# Load the yeast dataset
data <- mlr3data::yeast
head(data)
```

The dataset contains several numeric features and multiple binary label columns. Each label column represents whether a particular class is present (1) or absent (0).

---

## Step 3: Prepare the Data  

Convert categorical variables (if any) and define the task:

```r
# Define the task for multi-label classification
task <- TaskMultioutput$new(id = "yeast", backend = data, 
                            target = colnames(data)[(ncol(data) - 13):ncol(data)]) # Selecting last 14 columns as labels

print(task)
```

---

## Step 4: Train a Multi-Label Model  

We will train a Random Forest model for multi-label classification.

```r
# Define the learner (Random Forest)
learner <- lrn("classif.ranger", predict_type = "prob")

# Train the model
learner$train(task)
```

---

## Step 5: Make Predictions  

```r
# Predict on the dataset
predictions <- learner$predict(task)

# View predictions
print(predictions)
```

---

## Step 6: Evaluate Model Performance  

Common evaluation metrics for multi-label classification include:  
- **Hamming Loss** (lower is better)  
- **Accuracy** (higher is better)  
- **F1 Score** (higher is better)  

```r
# Compute performance metrics
hamming_loss <- mlr3measures::hamming_loss(predictions$score(), data[, (ncol(data) - 13):ncol(data)])
accuracy <- mlr3measures::accuracy(predictions$score(), data[, (ncol(data) - 13):ncol(data)])
f1_score <- mlr3measures::f1(predictions$score(), data[, (ncol(data) - 13):ncol(data)])

# Print results
cat("Hamming Loss:", hamming_loss, "\n")
cat("Accuracy:", accuracy, "\n")
cat("F1 Score:", f1_score, "\n")
```

---

## Step 7: Hyperparameter Tuning (Optional)  

We can optimize the model’s hyperparameters to improve performance.  

```r
# Define parameter tuning space
param_set <- ParamSet$new(list(
  ParamInt$new("num.trees", lower = 50, upper = 500)
))

# Define a tuning instance
tuner <- TunerGridSearch$new()
instance <- TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 5),
  measure = msr("hamming_loss"),
  param_set = param_set,
  terminator = trm("evals", n_evals = 10),
  tuner = tuner
)

# Run tuning
tuner$optimize(instance)

# Best hyperparameters
print(instance$result)
```

---

## Conclusion  

In this tutorial, we covered the basics of multi-label classification in R using `mlr3`. We:  
✔ Loaded and prepared a dataset  
✔ Trained a Random Forest model  
✔ Evaluated performance using multi-label metrics  
✔ Optimized hyperparameters  

You can further experiment with different models (`classif.rpart`, `classif.svm`, etc.) and feature engineering techniques to improve performance. 🚀
