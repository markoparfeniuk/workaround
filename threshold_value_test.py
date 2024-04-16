import spacy
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Load spaCy's English language model
nlp = spacy.load('en_core_web_lg')

# Check if two words are synonyms based on a similarity threshold.
def is_synonym(word1, word2, threshold):
    return nlp(word1).similarity(nlp(word2)) > threshold

# Evaluate how a specific threshold performs in identifying synonyms.
def evaluate_threshold(word_synonym_pairs, threshold):
    true_labels = [pair[2] for pair in word_synonym_pairs]
    predictions = [is_synonym(pair[0], pair[1], threshold) for pair in word_synonym_pairs]
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    return precision, recall, f1

# Assuming word_synonym_pairs is loaded from your file as before
def find_optimal_threshold(word_synonym_pairs, accuracy=5):
    left, right = 0.0, 1.0
    iteration = 0

    while (right - left) > 10**(-accuracy):
        thresholds = np.linspace(left, right, 10)
        scores = [evaluate_threshold(word_synonym_pairs, t)[2] for t in thresholds]
        max_index = np.argmax(scores)  # Index of the maximum F1 score

        # Print detailed scores for the current range
        print(f"\nIteration {iteration+1} detailed scores within range [{left:.5f}, {right:.5f}]:")
        for i, threshold in enumerate(thresholds):
            precision, recall, f1 = scores[i], scores[i], scores[i]  # For simplification
            print(f"Threshold: {threshold:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1 Score: {f1:.5f}")

        # Adjust the range based on the position of the max_index
        if max_index == 0:
            left, right = thresholds[max_index], thresholds[max_index + 1]
        elif max_index == len(thresholds) - 1:
            left, right = thresholds[max_index - 1], thresholds[max_index]
        else:
            left, right = thresholds[max_index - 1], thresholds[max_index + 1]

        iteration += 1
        print(f"\nAfter Iteration {iteration}: Range narrowed to [{left:.5f}, {right:.5f}], with max F1 Score: {scores[max_index]:.5f} at Threshold: {thresholds[max_index]:.5f}")

    optimal_threshold = thresholds[max_index]
    return optimal_threshold

# Read word-synonym pairs from file
word_synonym_pairs = []
with open('word_synonym_pairs.txt', 'r') as file:
    for line in file:
        word1, word2, label = line.strip().split(',')
        word_synonym_pairs.append((word1, word2, label == 'True'))


# Example usage
optimal_threshold = find_optimal_threshold(word_synonym_pairs)
print(f"Optimal Threshold: {optimal_threshold:.5f}")