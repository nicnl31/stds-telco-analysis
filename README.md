# Statistical Analysis and Modelling on a Telecommunications Dataset

## Introduction

This project aims to analyse a telecommunications dataset to identify customers most likely to subscribe to a new data plan, and to give expert information to Management and team to execute the next marketing campaign.

The study includes the use of four statistical models that compete to get the best metric score in a cross-validation setting. These models are chosen from a mix of parametric (**P**), non-parametric (**NP**), and ensembles (**E**):
- Logistic Regression (LR) (**P**)
- k-Nearest Neighbours (kNN) (**NP**)
- Decision Trees (DT) (**P**)
- Random Forest (RF) (**E**)

## How to use this repository
The main codebase for this repository is available in the `code/` directory, where:
- The `EDA` notebook explores the data
- The `Report` notebook builds statistical models to for prediction

Detailed reports for each phase of the study is available in the `report/` repository.

## The data
The data is available in the `code/` directory, which contains de-identified customer information from a telecommunications company in a single quarter. It includes:
- Demographic variables e.g. age, job, education status
- Macro-economic factors e.g. Euribor rate.

The target is `y`, which denotes whether or not the customer subscribed to a new communication plan.


## The statistical models

### Logistic Regression
Logistic Regression (LR) was chosen because of its straightforward interpretation of coefficients. Lasso-penalised LR is used to prevent overfitting. The goal is to minimise the Lasso negative log-likelihood via maximum likelihood estimation:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^n \big[- y_i \ln(\hat{p}(\mathbf{x}_i)) - (1-y_i)\ln(1 - \hat{p}(\mathbf{x}_i))\big] + \lambda\Vert\boldsymbol{\beta}\Vert_1,$$

where $\hat{p}(\mathbf{x}_i) = 1\big/(1 + \exp{-(\mathbf{x}_i^\top \boldsymbol{\beta} + \beta_0)})$ is the estimated probability that the observation $\mathbf{x}_i$ belongs to the positive class.

### k-Nearest Neighbours
k-nearest neighbours (kNN) was chosen because of its simplicity and ease of interpretation. Its goal is to predict a class based on neighbour data points, by finding the class $c$ that maximises the number of neighbour points $\mathbf{x}_j$ in a set $S_k$ of $k$ neighbours:

$$\hat{y_i} = \argmax_c \sum_{j:\: \mathbf{x}_j \in  S_k}\mathbf{1}\{\mathbf{x}_j \text{ in class } c \}$$

### Decision Tree and Random Forest
Decision tree (DT) was also considered in addition because of its computational speed,
while also providing a base for training ensemble algorithms. 

Random Forest (RF) was built as an ensemble of multiple DT’s to enhance the overall predictive power, whilst reducing the effect of overfitting by the individual trees. It predicts a
class based on the majority vote of $n$ DT’s, where $n$ is to be determined via cross-validation.