# Predicting Policy Rate Changes from Central Bank Minutes Using Machine Learning  
**Evidence from the Czech Republic (1998–2024)**  

This repository contains the code and resources used for my Master's thesis. It explores the use of transfer learning to predict policy rate changes based on sentiment extracted from the Czech National Bank (CNB) minutes.  

## Repository Structure  

### Key Scripts  
- **`cbminutes_dataset.py`**  
  Prepares the data for training, development, and testing.  
  Builds on code from [this repository](https://github.com/ufal/npfl138?tab=readme-ov-file).  

- **`sentiment_analysis.py`**  
  - Defines the model architecture.  
  - Processes the data.  
  - Fine-tunes the chosen language model for sentiment analysis.  
  Builds on code from [this repository](https://github.com/ufal/npfl138?tab=readme-ov-file).  

- **`predict_sentiments.py`**  
  Uses the fine-tuned model to predict the sentiment of the CNB minutes.  

- **`trainable_module.py`**  
  Provides the necessary infrastructure to train the model in a Keras-like way.  
  Builds on code from [this repository](https://github.com/ufal/npfl138?tab=readme-ov-file).

- **`run_experiments.sh`**  
  Shell script that allows running experiments with different hyperparameter configurations.
  
### Data  
- **`annotated_minutes.csv`**  
  Annotated text sourced from Nitoi, Pochea, and Radu (2023).  

- **`cnb_minutes/`**  
  Directory containing the CNB minutes, sourced from the Czech National Bank (2024).  

## Prerequisites  
Install the required Python packages listed in `requirements.txt`.  

## References  
1. **Nitoi, M., Pochea, M.-M., & Radu, Ş.-C.** (2023).  
   *Unveiling the sentiment behind central bank narratives: A novel deep.*  
   Journal of Behavioral and Experimental Finance, 38.  

2. **Czech National Bank.** (2024).  
   *Bank Board decisions.*  
   Retrieved December 20, 2024, from the [Czech National Bank website](https://www.cnb.cz/en/monetary-policy/bank-board-decisions/).  

## Acknowledgments  
This project utilizes code from the [NPFL138 repository](https://github.com/ufal/npfl138?tab=readme-ov-file).  
