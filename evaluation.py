import argparse
import json
import pickle
from tqdm import tqdm
from pathlib import Path
import re

def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process benchmark JSON and calculate accuracy.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file to be evaluated')
    
    args = parser.parse_args()  
    
    with open(args.input, 'r') as f:
        input_data = json.load(f)

    corr, total = 0, 0


    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}

    output_key = 'model_prediction' # The key that contains model output
    no_pred_count = 0
    matched_outputs = []
    new_data = []
    for idx, sample in enumerate(tqdm(input_data)):

        if sample['split'] == 'test':
            continue

        if output_key not in sample:
            continue

        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        task = sample['task']
        difficulty = sample['difficulty']
        choices = sample['choices']

        if string_match(_answer, _prediction, choices):
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        total += 1
        new_data.append(sample)
        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1


    print("*"*30)
    for task in task_metrics:
    
        print(f"{task} : {(task_metrics[task][0]/task_metrics[task][1])*100 if task_metrics[task][1] != 0 else 0} over {task_metrics[task][1]} samples")
    print("*"*30)
    for diff in diff_metrics:
        print(f"{diff} : {(diff_metrics[diff][0]/diff_metrics[diff][1])*100}")
    print("*"*30)
    print(f"Total acc: {(corr/total) * 100} over {total} samples")
    print("*"*30)
    print(f"No prediction count: {no_pred_count}")
