# Spam Filter in Rust

A simple Naive Bayes classifier implemented in Rust for detecting spam emails based on word frequency. 
A project for recreastional purposes, inspired by the need to understand machine learning concepts and Rust programming.


## Overview

This project uses a Naive Bayes algorithm to classify emails as either **Spam** or **Ham** (non-spam). It processes a dataset of word counts extracted from emails, trains on a portion of the data, and evaluates its performance on the remainder.
It uses the data from kaggle: [Kaggle dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data) and is designed to be efficient and easy to understand, making it a great starting point for anyone interested in machine learning with Rust.

## Features

- **Naive Bayes Classification**: Uses probabilistic modeling for classification.
- **Laplace Smoothing**: Handles words not seen during training to avoid zero-probability issues.
- **Performance Metrics**: Calculates Accuracy, Precision, Recall, and displays a Confusion Matrix.
- **CSV Data Processing**: Efficiently reads and parses large datasets using the `csv` crate.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (2024 edition or later)
- Cargo (Rust's package manager)

## Project Structure

- `src/main.rs`: The main implementation of the classifier and evaluation logic.
- `data/emails.csv`: The dataset containing word frequencies and labels.
- `Cargo.toml`: Project dependencies and configuration.

## Dataset

The project expects a CSV file at `data/emails.csv` with the following format:
- First column: Email identifier (e.g., "Email 1").
- Middle columns: Word frequency counts for several thousand common words.
- Last column (`Prediction`): `0` for ham, `1` for spam.

## Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd spam-filter
   ```


3. **Run the classifier**:
   ```bash
   cargo run --release
   ```

## How it Works

1. **Loading Data**: The application reads `data/emails.csv` and splits it into two equal halves for training and testing.
2. **Training**:
   - Calculates the log-prior probabilities for spam and ham classes.
   - Calculates the log-likelihood for each word given each class, using Laplace smoothing ($+1$ to counts).
3. **Classification**:
   - For each test email, it computes the log-score for both spam and ham classes using the Naive Bayes formula.
   - The class with the higher score is chosen as the prediction.
4. **Evaluation**:
   - Compares predictions against actual labels.
   - Outputs the model's accuracy, precision, and recall.
