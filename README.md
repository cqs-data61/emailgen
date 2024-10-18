# Generating Deceptive Social Interactions

Python implementation of the paper ["Modelling Direct Messaging Networks with Multiple Recipients for Cyber Deception"](https://ieeexplore.ieee.org/document/9797379/), EuroS&P 2022.

The LogNormMix-Net temporal point process (TPP) is used to model the temporal dynamics of email communication, and the ["Huggingface implementation of GPT2"](https://huggingface.co/gpt2) is used for generating the email text.


## Usage
The Conda environment (Linux) that was used in this work is included here: [`emailgen.yml`](https://bitbucket.csiro.au/projects/DECAAS/repos/emailgen-draft/browse/emailgen.yml). A more general requirements file can be found here: [`code/requirements.txt`](https://bitbucket.csiro.au/projects/DECAAS/repos/emailgen-draft/browse/code/requirements.txt). 

To generate email content from our pre-trained demo model:
Run the notebook [`code/Run-email-generation.ipynb`](https://bitbucket.csiro.au/projects/DECAAS/repos/emailgen-draft/browse/code/Run-email-generation.ipynb) to generate timestamps, sample email sender/recipients, and generate email text based on the underlying distribution of the training data.


To train a model on your own data:

1. Data:  
The TPP datasets from the paper are found in the [`data/`](https://bitbucket.csiro.au/projects/DECAAS/repos/emailgen-draft/browse/data/) directory. Details on how to format the data for training the LogNormMix-Net are given below.  
The Enron dataset used to train the language model can be found here: https://www.cs.cmu.edu/~./enron/

2. Train the TPP:  
The Jupyter notebook [`code/train_TPP.ipnb`](https://bitbucket.csiro.au/projects/DECAAS/repos/emailgen-draft/browse/code/train_TPP.ipynb) is used to train the LogNormMix-Net TPP. 

3. Train the language model:  
We have included the script from Huggingface/transformers to fine-tune a GPT-2 model: [`code/run_language_modeling.py`](https://bitbucket.csiro.au/projects/DECAAS/repos/emailgen-draft/browse/code/run_language_modeling.py).

To find tune GPT2 model on your own dataset, run 

```bash
python run_language_modeling.py \
--output_dir=models/gpt2_fine-tune \
--model_type=gpt2 \
--model_name_or_path=gpt2-finetuned \
--do_train \
--train_data_file='train_data.txt' \
--do_eval \
--eval_data_file='eval_data.txt' \
--per_device_train_batch_size=5 \
--per_device_eval_batch_size=5 \
--line_by_line \
--evaluate_during_training \
--learning_rate 5e-5 \
--num_train_epochs=5
```

We have also included the script from Huggingface/transformers to generate text from your fine-tuned model: [`code/run_generation.py`](https://bitbucket.csiro.au/projects/DECAAS/repos/emailgen-draft/browse/code/run_generation.py).

To generate content from your fine-tuned model, run:
```bash
python run_generation.py  \
--model_type gpt2  \
--model_name_or_path models/gpt2-finetuned/   
--length 150  \
--prompt "FYI. The report posted on "   \
--num_return_sequences 3
```
