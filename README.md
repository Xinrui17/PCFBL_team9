# PCFBL_team9

.ipynb is available in /code directory

## How to Run the Pipeline

### Running Prediction Only
To execute predictions only, simply run the notebook from the beginning without making any changes. This will use the pre-trained model to perform predictions as outlined in the notebook steps.

### Running the Training Process
If you wish to train the model yourself, you will need to modify the code in Section 4 of the notebook:
1. Locate the `main()` function.
2. Uncomment the line `run_private_clustering(MODELTYPE)`.

   This will activate the training process using the specified `MODELTYPE`.
3. Run the notebook starting from Section 4 to initiate training.

Follow these steps to customize the execution of the pipeline according to your needs.
