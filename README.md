# Text Generator using PyTorch Lightning

This project is a text generator built using PyTorch Lightning. It leverages the BookCorpus dataset from Hugging Face and a custom tokenizer to generate text based on the input data.

## Project Structure

text-generator/<br>
├── pycache/<br>
├── checkpoints/<br>
├── lightning_logs/<br>
├── app.py<br>
├── bookcorpus.txt<br>
├── data.py<br>
├── model.py<br>
├── tokenizer.json<br>
├── requirements.txt<br>
└── README.md<br>


## Dataset

The dataset used in this project is the BookCorpus dataset from Hugging Face. The BookCorpus dataset is a large-scale dataset consisting of books written by unpublished authors. It contains 11,038 books and is used for various NLP tasks such as language modeling, text generation, and more.

For more information on the BookCorpus dataset, visit the [Hugging Face BookCorpus page](https://huggingface.co/datasets/bookcorpus).

## Installation

To set up the environment and install the required dependencies, follow these steps:
**Use Anaconda Platform**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhuvi-ai/Text-Generator-Using-Pytorch.git
   cd Text-Generator-Using-Pytorch
2. **Create and activate a Conda environment:**
     ```bash
   conda create -n text-generator python=3.10.0
   conda activate text-generator
3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt

**How to Run**
1. Prepare the dataset:
   using data.py preprepare daaset by donwloading and preprocessing
   ```bash
   python data.py

Ensure that the bookcorpus.txt file is placed in the project directory. This file should contain the preprocessed text data from the BookCorpus dataset.<br>
2. Train the model:<br>
  To train the text generator model, run the following command<br>
   ```bash
    python model.py
   ```
3. **Run the Streamlit app:**
To launch the Streamlit app and interact with the text generator, run:
  ```bash
  streamlit run app.py
  ```

**Explanation of Files**
> __pycache__/: Directory containing Python bytecode files.<br>
> checkpoints/: Directory where model checkpoints are saved during training.<br>
> lightning_logs/: Directory where PyTorch Lightning logs are stored.<br>
> app.py: Streamlit app for interacting with the text generator.<br>
> bookcorpus.txt: Preprocessed text data from the BookCorpus dataset.<br>
> data.py: Script for data preprocessing and loading.<br>
> model.py: Script defining the text generator model and training process.<br>
> tokenizer.json: Custom tokenizer configuration file.<br>
> requirements.txt: File containing the list of dependencies required for the project.<br>

