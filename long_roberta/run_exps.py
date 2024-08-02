import os  
import subprocess  
import yaml  
  
def update_params_yaml(
        params,
        dataset_path,
        output_path,
        model_name,
        pooling_strategy,
        tokenizer_path,
        bert_path
    ):  
    params['dataset_path'] = dataset_path  
    params['output_path'] = output_path  
    params['model_name'] = model_name  
    params['pooling_strategy'] = pooling_strategy  
    params['tokenizer_path'] = tokenizer_path  
    params['bert_path'] = bert_path
  
    with open('params.yml', 'w') as params_file:  
        yaml.dump(params, params_file) 

def run_train_and_evaluate():  
    subprocess.run(['python', 'train_and_evaluate.py'])  

# Load the params.yml file  
with open('params.yml', 'r') as params_file:  
    params = yaml.safe_load(params_file)
    
# Set tokenizer_path and bert_path  
tokenizer_path = 'roberta_v2/'
bert_path = 'checkpoint-500000/' 

datasets = [
    '_text_label.hf',
    '_text_label.hf',
    '_text_label.hf',
    '_text_label.hf',
    '_text_label.hf',
    '_text_label.hf'
]  
output_paths = [
    '_text_only/',
    '_text_only_mean/',
    '_text_only/',
    '_text_only_mean/',
    '_text_only/',
    '_text_only_mean/'
]  
model_names = [
    "_text_only",
    "_text_only_mean",
    "_text_only",
    "_text_only_mean",
    "_text_only",
    "_text_only_mean"
]  
pooling_strategies = [
    "max",
    "mean",
    "max",
    "mean",
    "max",
    "mean"
]  

assert len(datasets) == len(output_paths) == len(model_names) == len(pooling_strategies), "Error in param lists."

# Iterate through the different values and run train_and_evaluate.py  
for dataset, output_path, model_name, pooling_strategy in zip(datasets, output_paths, model_names, pooling_strategies):
    
    # Update params.yml file with new values  
    update_params_yaml(
        params,
        dataset,
        output_path,
        model_name,
        pooling_strategy,
        tokenizer_path,
        bert_path
    )  

    # Create output directory if it doesn't exist  
    os.makedirs(output_path, exist_ok=True)  

    # Run the train_and_evaluate.py script  
    run_train_and_evaluate()  
