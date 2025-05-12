'''
This script implements the CHARM Algorithm 1 from https://arxiv.org/html/2504.10045v1 (Yu et al, 2025). It takes as input save locations, ELO scores, and model names for a generative model and a reference model and generates a calibrated dataset.
'''

# Imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from datasets import load_dataset
import random
from huggingface_hub import login
from vllm import LLM, SamplingParams

# SETUP: Save locations:
prompt_save_location = ""
calibrated_score_save_location = ""
calibrated_data_save_location = ""
uncalibrated_data_save_location = ""

generative_tokenizer_save_location = ""
generative_model_save_location = ""

reference_tokenizer_save_location = ""
reference_model_save_location = ""

reward_tokenizer_save_location = ""
reward_model_save_location = ""

# Models Used
gen_model_name = ""
ref_model_name = ""
reward_model_name = ""

# Change if part of the pipeline has worked
prompts_generated = False
uncorrected_dataset_generated = False
corrected_dataset_generated = False
generative_model_saved = False
reference_model_saved = False
reward_model_saved = False

# ELO scores
gen_elo = 
ref_elo = 

# Huggingface Login Key
login("")

# Expected Probability (Equation 2)
def expect_prob(ELO_O, ELO_R):
  return 1 / (1 + 10 ** ((ELO_R - ELO_O) / 400))

def make_prompt_dataset(num_samples)
    dataset = load_dataset("hendrydong/preference_700K", split="train", streaming=True)

    D = []

    random.seed(42)

    for i, sample in enumerate(dataset):
        if i >= 200:
            break

        x_i = sample['rejected'][0]['content']
        D.append(x_i)

    print(f"Collected {len(D)} triplets.")

    df = pd.DataFrame(D, columns=["prompt"])

    df.to_csv(prompt_save_location, index=False)

    print(f"Saved dataset to {prompt_save_location}")
    return df

# Empirical Probability (Equation 3)
def empirical_prob(s_iO, s_iR):
  return torch.mean(torch.sigmoid(s_iO - s_iR))

# Calculate Offset Gradient to minimize delta
def update_Grad(T, expected_prob, s_iO, s_iR, learning_rate):
    grad = 0.0
    s_iO = torch.tensor(s_iO.values, dtype=torch.float32)
    s_iR = torch.tensor(s_iR.values, dtype=torch.float32)
    expected_prob = torch.full((len(s_iO),), expected_prob)

    delta = s_iO + grad - s_iR
    p_hat = torch.sigmoid(delta) 

    grad_tensor = 2 * (p_hat - expected_prob) * p_hat * (1 - p_hat)

    return grad_tensor

# Apply Offset
def apply_offset(s_iO, offset):
  return torch.tensor(s_iO.values, dtype=torch.float32) + offset

def construct_calibrated_dataset(df, ELO_O, ELO_R):
    '''
    Inputs:
        df: Pandas Dataset columns=["prompt", "y_iO", "y_iR", "s_iO", "s_iR"]
        ELO_o: generative model ELO score
        ELO_r: reference model ELO score
    Outputs: 
        calibrated_df: Pandas Dataset D = {x, y+, y-}
        df: Pandas Dataset columns=["prompt", "y_iO", "y_iR", "s_iO", "s_iR", "s_iO_calibrated"]
        Note: change save locations above depending on where this is running
    '''

    expected_prob = expect_prob(ELO_O, ELO_R)

    # Grad hyperparameters
    T = 100
    learning_rate = 0.01
    grad = update_Grad(T, expected_prob, df['s_iO'], df['s_iR'], learning_rate)

    s_iO_calibrated = apply_offset(df['s_iO'], grad)

    df['s_iO_calibrated'] = s_iO_calibrated
    df.to_csv(calibrated_score_save_location, index=False)

    calibrated_dataset = []
    for el in df:
        if el['s_iO_calibrated'] > el['s_iR']:
            calibrated_dataset.append([el['prompt'], el['y_iO'], el['y_iR']])
        else:
            calibrated_dataset.append([el['prompt'], el['y_iR'], el['y_iO']])

    calibrated_df = pd.DataFrame(calibrated_dataset, columns=["prompt", "chosen", "rejected"])
    calibrated_df.to_csv(calibrated_data_save_location, index=False)

    return calibrated_df, df

def score_response(model, tokenizer, prompt, response):
    device = model.device
    text = [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}]

    conv_tokenized = tokenizer.apply_chat_template(text, tokenize=False, return_tensors="pt").to(device)

    kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
    tokens = tokenizer.encode_plus(conv_tokenized, **kwargs)

    with torch.no_grad():
        reward_tensor = model(tokens["input_ids"][0].view(1,-1).to(device), attention_mask=tokens["attention_mask"][0].view(1,-1).to(device))[0]
        score = reward_tensor.cpu().detach().item()
    
    return score

def gen_response_pipelined(model, prompt, max_new_tokens = 50):
    print(f"Pipelined Generation: {model}")
    generator = pipeline("text-generation", model=model)
    response = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]['generated_text']
    print(f"Response: {response}")
    return response

def gen_response_classic(model, tokenizer, prompt, max_new_tokens = 50):
    print(f"Generating Classic Response: {model.name_or_path}")
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    return response

# Might need to be modified to add tokenizer translation of prompt
def gen_response_llm(model, tokenizer, prompt, sampling_params= SamplingParams(temperature=0.7, top_p=0.8, max_tokens=40)):
    print(f"Generating LLM Response: {model.name_or_path}")
    outputs = model.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text

def build_competitive_dataset(prompts, reference_model, reference_tokenizer, gen_model, gen_tokenizer, reward_model, reward_tokenizer, pipelined = 1):
    '''
    Inputs:
        prompts: pd dataframe / array of prompts
        reference_model: name of reference model (for pipelined output) or preloaded reference model itself
        reference_tokenizer: none if pipelining output, otherwise pass preloaded reference_tokenizer
        gen_model: name of generative model (for pipelined output) or preloaded generative model itself
        gen_tokenizer: none if pipelining output, otherwise pass preloaded gen_tokenizer
        reward_model: preloaded model for scoring responses
        reward_tokenizer: preloaded tokenizer for RM
        pipelined: 1, 2, or 3 - indicates what type of response generation this is using
            1 = using transformers pipeline
            2 = using traditional transformers call
            3 = using llm call
    Outputs:
        gen_df: pd dataframe of generated and reference responses with associated scores: columns=["prompt", "y_iR", "y_iO", "s_iR", "s_iO"]
    '''

    gen_df = []
    for prompt in prompts:
        if pipelined == 1:
            reference_response = gen_response_pipelined(reference_model, prompt)
            generated_response = gen_response_pipelined(gen_model, prompt)
        elif pipelined == 2:
            reference_response = gen_response_classic(reference_model, reference_tokenizer, prompt)
            generated_response = gen_response_classic(gen_model, gen_tokenizer, prompt)
        elif pipelined == 3:
            reference_response = gen_response_llm(reference_model, reference_tokenizer, prompt)
            generated_response = gen_response_llm(gen_model, gen_tokenizer, prompt)
        else:
            print("Return valid integer (1, 2, or 3) for pipelined")
            return

        reference_score = score_response(reward_model, reference_tokenizer, prompt, reference_response)
        generated_score = score_response(gen_model, gen_tokenizer, prompt, generated_response)

        gen_df.append([prompt, reference_response, generated_response, reference_score, generated_score])
    
    df = pd.DataFrame(gen_df, columns=["prompt", "y_iR", "y_iO", "s_iR", "s_iO"])
    df.to_csv(uncalibrated_data_save_location, index=False)

    return df

def main():
    # Hyperparameters
    num_samples = 2000

    # If 1: using transformers pipeline, if 2: using transformers imports, if 3: using vllm library
    pipelined = 1

    # Load Models
    if pipelined == 1:
        print(f"Pipelined: Not loading generative and reference models")
        generative_model = gen_model_name
        generative_tokenizer = None

        reference_model = ref_model_name
        reference_tokenizer = None

    elif pipelined == 2:
        print(f"Loading Generative Tokenizer: {gen_model_name}")
        generative_tokenizer = AutoTokenizer.from_pretrained(gen_model_name) if not generative_model_saved else AutoTokenizer.from_pretrained(generative_tokenizer_save_location)
        print(f"Loading Generative Model: {gen_model_name}")
        generative_model = AutoModelForSequenceClassification.from_pretrained(gen_model_name) if not generative_model_saved else AutoModelForCausalLM.from_pretrained(generative_model_save_location)

        print(f"Loading Reference Tokenizer: {ref_model_name}")
        reference_tokenizer = AutoTokenizer.from_pretrained(ref_model_name) if not reference_model_saved else AutoTokenizer.from_pretrained(reference_tokenizer_save_location)
        print(f"Loading Reference Model: {ref_model_name}")
        reference_model = AutoModelForSequenceClassification.from_pretrained(ref_model_name) if not reference_model_saved else AutoModelForCausalLM.from_pretrained(reference_model_save_location)

    elif pipelined == 3:
        print(f"Loading Generative Tokenizer: {gen_model_name}")
        generative_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        print(f"Loading Generative Model: {gen_model_name}")
        generative_model = LLM(model=gen_model_name, tensor_parallel_size=1)

        print(f"Loading Reference Tokenizer: {ref_model_name}")
        reference_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
        print(f"Loading Reference Model: {ref_model_name}")
        reference_model = LLM(model=ref_model_name, tensor_parallel_size=1)

    else:
        print("Error: Please Return valid integer (1, 2, or 3) for pipelined")

    print(f"Loading Reward Tokenizer: {reward_model_name}")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name) if not reward_model_saved else AutoTokenizer.from_pretrained(reward_tokenizer_save_location)
    print(f"Loading Reward Model: {reward_model_name}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name) if not reward_model_saved else AutoModelForCausalLM.from_pretrained(reward_model_save_location)

    # Save loaded models
    if not generative_model_saved and pipelined == 2:
        print(f"Saving Generative Model: {generative_model_save_location}")
        generative_model.save_pretrained(generative_model_save_location)
        print(f"Saving Generative Tokenizer: {generative_model_save_location}")
        generative_tokenizer.save_pretrained(generative_tokenizer_save_location)
    
    if not reference_model_saved and pipelined == 2:
        print(f"Saving Reference Model: {reference_model_save_location}")
        reference_model.save_pretrained(reference_model_save_location)
        print(f"Saving Reference Tokenizer: {reference_model_save_location}")
        reference_tokenizer.save_pretrained(reference_tokenizer_save_location)

    if not reward_model_saved:
        print(f"Saving Reward Model: {reward_model_save_location}")
        reward_model.save_pretrained(reward_model_save_location)
        print(f"Saving Reward Tokenizer: {reward_model_save_location}")
        reward_tokenizer.save_pretrained(reward_tokenizer_save_location)

    # Check for previously generated elements, either load or generate each
    if not prompts_generated:
        print("Generating Prompts")
        prompt_dataset = make_prompt_dataset(num_samples)
    else:
        print(f"Finding Prompts at {prompt_save_location}")
        prompt_dataset = pd.read_csv(prompt_save_location)

    if not uncorrected_dataset_generated:
        print("Generating Uncorrected Dataset")
        uncorrected_dataset = build_competitive_dataset(prompt_dataset, reference_model, reference_tokenizer, generative_model, generative_tokenizer, reward_model, reward_tokenizer, pipelined)
    else:
        print(f"Finding Uncorrected Dataset at {uncalibrated_data_save_location}")
        uncorrected_dataset = pd.read_csv(uncalibrated_data_save_location)
    
    if not corrected_dataset_generated:
        print("Generating Corrected Dataset")
        corrected_dataset = construct_calibrated_dataset(uncorrected_dataset, gen_elo, ref_elo)
    else:
        print(f"Finding Corrected Dataset at {calibrated_data_save_location}")
        corrected_dataset = pd.read_csv(calibrated_data_save_location)

    print(corrected_dataset)
    print("Done")

if __name__ == "__main__":
    main()
        

