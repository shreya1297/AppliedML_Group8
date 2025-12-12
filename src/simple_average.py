import torch
import torch.nn as nn
from collections import Counter


class SimpleAverageModel(nn.Module):
    """
    A trivial model used to test the pipeline.

    It completely ignores the text and always predicts
    the most frequent label found in the training dataset.
    """

    def __init__(self, most_common_label: int):
        super().__init__()
        self.most_common_label = most_common_label

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Return logits that strongly prefer the majority label.
        """
        batch_size = input_ids.shape[0]
        num_choices = 4  # Each question has 4 options

        # Zero logits + strong positive score for majority class
        logits = torch.zeros(batch_size, num_choices)
        logits[:, self.most_common_label] = 5.0

        return logits


def compute_most_common_label(dataset):
    """
    Given a dataset with a 'label' column,
    return the integer label that appears most often.
    """
    labels = [int(x) for x in dataset["label"]]
    counter = Counter(labels)
    most_common_label, _ = counter.most_common(1)[0]
    return most_common_label
