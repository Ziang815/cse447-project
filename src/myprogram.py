#!/usr/bin/env python
import os
import string
import random
import torch
from transformers import BertTokenizer, BertForMaskedLM
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class MyModel:
    """
    BERT-based model to predict the next character in a string.
    """

    def __init__(self):
        # Load the pre-trained mBERT model and tokenizer
        self.model_name = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForMaskedLM.from_pretrained(self.model_name)
        # self.model.eval()  # Set model to evaluation mode

    @classmethod
    def load_training_data(cls):
        # No training needed for checkpoint 1
        return []

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                inp = line.strip()  # Remove newline characters
                if inp:
                    data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # No training needed for checkpoint 1
        pass

    def run_pred(self, data):
        """
        Predict the next character for each input string using BERT.
        """
        predictions = []

        with torch.no_grad():  # Disable gradient calculations
            for sentence in data:
                # Prepare input for BERT by adding a space between characters and appending [MASK]
                char_token = " ".join(sentence) + " [MASK]"
                inputs = self.tokenizer(char_token, return_tensors="pt")

                # Find the index of the [MASK] token
                mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1].item()

                # Get model predictions
                outputs = self.model(**inputs)

                # Get the top 3 predicted token IDs
                top_3_predictions = torch.topk(outputs.logits[0, mask_token_index], k=3)

                # Convert token IDs back to characters
                predicted_chars = [self.tokenizer.decode([token_id]).strip() for token_id in top_3_predictions.indices]

                # Append to predictions list as a string of 3 characters
                predictions.append(''.join(predicted_chars))

        return predictions

    def save(self, work_dir):
        # Save a dummy checkpoint file for demonstration
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # Load the model from the pre-trained checkpoint (no actual checkpoint loading for now)
        print("Loading pre-trained BERT model...")
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Creating working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training skipped for checkpoint 1')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model...')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions...')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
        print('Prediction completed successfully!')
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))