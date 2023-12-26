# language-model-based-patient-embedding
Code for the language-model-based-patient-embedding paper

## Get started
if using https:
``````
git clone https://github.com/suxian06/language-model-based-patient-embedding.git
``````
or SSH:
``````
git@github.com:suxian06/language-model-based-patient-embedding.git
``````

## Demo using dummy data from the example_data folder

The example_data folder contains synthetic dummy data that can be used to demo
the model. The detailed model architecture can be found in
![alt text](https://https://github.com/suxian06/language-model-based-patient-embedding/blob/main/model_architecture.png?raw=true)
### Run the Autoencoder models (embedding of vocabularies)
``````
python scripts/autoencoder.py

``````

### Run the Transformer models (embedding patient vectors)

``````
python scripts/transformer.py

``````
### Run the Patient Embedding models (adopted from S-BERT)

``````
python scripts/patembd.py

``````
