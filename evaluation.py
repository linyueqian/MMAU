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

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    import json
    from tqdm import tqdm
    import re

    print("Starting Evaluation.....")
    output = {}

    # Read the test data and user submission
    with open(test_annotation_file, 'r') as f:
        input_data = json.load(f)

    with open(user_submission_file, 'r') as f:
        user_predictions = json.load(f)

    # Check if the number of predictions matches the number of samples
    if len(input_data) != len(user_predictions):
        raise ValueError("Number of predictions does not match number of samples.")

    # Attach predictions to the input data
    for idx, sample in enumerate(input_data):
        sample['model_prediction'] = user_predictions[idx]['model_prediction']

    # Initialize variables
    splits = set(sample['split'] for sample in input_data)
    split_metrics = {}
    for split in splits:
        split_metrics[split] = {
            'corr': 0,
            'total': 0,
            'task_metrics': {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]},
            'diff_metrics': {'easy': [0, 0], 'medium': [0, 0], 'hard': [0, 0]},
            'Total': 0,
            'no_pred_count': 0
        }

    output_key = 'model_prediction'
    for idx, sample in enumerate(tqdm(input_data)):
        split = sample['split']

        if output_key not in sample or not sample[output_key]:
            _prediction = ''
            split_metrics[split]['no_pred_count'] += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        task = sample['task']
        difficulty = sample['difficulty']
        choices = sample['choices']

        if string_match(_answer, _prediction, choices):
            split_metrics[split]['corr'] += 1
            split_metrics[split]['task_metrics'][task][0] += 1
            split_metrics[split]['diff_metrics'][difficulty][0] += 1
            sample['match'] = 1
        else:
            sample['match'] = 0
            # Debugging output for mismatches
            print(f"Mismatch at index {idx}:")
            print(f"Answer: '{_answer}'")
            print(f"Prediction: '{_prediction}'")
            print(f"Choices: {choices}")

        split_metrics[split]['total'] += 1
        split_metrics[split]['task_metrics'][task][1] += 1
        split_metrics[split]['diff_metrics'][difficulty][1] += 1

    # Prepare the output based on the phase
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        split = 'test-mini'
        if split in split_metrics:
            metrics = split_metrics[split]
            submission_result = {
                f"{task} Accuracy": (metrics['task_metrics'][task][0] / metrics['task_metrics'][task][1]) * 100
                if metrics['task_metrics'][task][1] != 0 else 0
                for task in metrics['task_metrics']
            }
            submission_result.update({
                f"{diff} Accuracy": (metrics['diff_metrics'][diff][0] / metrics['diff_metrics'][diff][1]) * 100
                if metrics['diff_metrics'][diff][1] != 0 else 0
                for diff in metrics['diff_metrics']
            })
            submission_result['Total'] = (metrics['corr'] / metrics['total']) * 100 if metrics['total'] != 0 else 0
            submission_result['No Prediction Count'] = metrics['no_pred_count']
            output["result"] = [{'test-mini': submission_result}]
            print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = []
        for split in ['test']:
            if split in split_metrics:
                metrics = split_metrics[split]
                submission_result = {
                    f"{task} Accuracy": (metrics['task_metrics'][task][0] / metrics['task_metrics'][task][1]) * 100
                    if metrics['task_metrics'][task][1] != 0 else 0
                    for task in metrics['task_metrics']
                }
                submission_result.update({
                    f"{diff} Accuracy": (metrics['diff_metrics'][diff][0] / metrics['diff_metrics'][diff][1]) * 100
                    if metrics['diff_metrics'][diff][1] != 0 else 0
                    for diff in metrics['diff_metrics']
                })
                submission_result['Total'] = (metrics['corr'] / metrics['total']) * 100 if metrics['total'] != 0 else 0
                submission_result['No Prediction Count'] = metrics['no_pred_count']
                output["result"].append({split: submission_result})
                print(f"Total Accuracy for {split}: {submission_result['Total']}")
        print("Completed evaluation for Test Phase")
    else:
        print(f"Unknown phase codename: {phase_codename}")
    print(output)
    return output
