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
4. Uncomment main(0) in the evaluation section

Follow these steps to customize the execution of the pipeline according to your needs.

## How to Run the Ablation Study

### Running Prediction Only
To execute predictions only, simply run the notebook from the beginning without making any changes. This will use the pre-trained model to perform predictions as outlined in the notebook steps.

### Running the Training Process
If you wish to train the model yourself, you will need to first modify the codes in /code directory:
1. Edit hyperparamter SUFFIX at the start of server.py, server_private.py, client_private.py, client_prediction.py, client_autoencoder.py.
   * To run the ablation study without autoencoder (directly calculating similarity with raw features), define SUFFIX as:
     SUFFIX = '_raw'
   * To run the ablation study without spectral clustering (replacing it with KMeans), define SUFFIX as:
     SUFFIX = '_kmeans'
2. Follow th same instructions for running the pipeline
