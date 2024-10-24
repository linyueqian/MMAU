import json
import pickle
from tqdm import tqdm
from pathlib import Path

def string_match(answer, prediction, choices):

    answer = str(answer)
    prediction = str(prediction)
    choices = [str(choice) for choice in choices]
    if len(prediction) == 0:
        return False

    not_answers = [choice.lower() for choice in choices if choice != answer]

    cond1 = answer.lower() in prediction.lower()
    cond2 = not any(choice.lower() in prediction.lower() for choice in not_answers)

    if cond1 and cond2:
        return True
    else:
        return False

if __name__ == "__main__":

    output_path = "JSON_OUTPUT_PATH"
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    corr, total = 0, 0


    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}

    output_key = 'model_output' # The key that contains model output
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
