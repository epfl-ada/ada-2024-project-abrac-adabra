import unittest
import random
import torch
from src.models.ProteinModel import ProteinModel
from transformers import EsmTokenizer, EsmModel


class Test(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test.
        """
        # Set the pipeline for comparison with the tokenizer
        self.model = ProteinModel(model="facebook/esm2_t6_8M_UR50D")

    def test_probabilities(self):
        """
        Test the probabilities of masked positions
        """
        masked_sequence = 'MQIFVKTLTGKTITLEVEPS<mask>TIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

        # Get the probabilities of the model
        predictions_model = self.model.get_probabilities_masked(masked_sequence)

        # Check from online probabilities
        for value in predictions_model:
            if value['token'] == 'D':
                self.assertAlmostEqual(value['score'], 0.322)
            if value['token'] == 'E':
                self.assertAlmostEqual(value['score'], 0.182)
            if value['token'] == 'A':
                self.assertAlmostEqual(value['score'], 0.108)
            if value['token'] == 'T':
                self.assertAlmostEqual(value['score'], 0.098)

    def test_predictions(self):
        """
        Test that the function correctly mask the sequence
        """
        sequence = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
        position = random.randint(0, len(sequence) - 1)
        masked_sequence = sequence[:position] + '<mask>' + sequence[position + 1:]

        predictions_model_masked = self.model.get_probabilities_masked(masked_sequence)
        predictions_model_unmasked = self.model.get_probabilities_unmasked(sequence, position)

        for row1, row2 in zip(predictions_model_masked, predictions_model_unmasked):
            self.assertEqual(row1['token'], row2['token'])
            self.assertEqual(row1['score'], row2['score'])

    def test_embeddings(self):
        """
        Test embeddings
        """
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        seqs = ["QERLKSIVRILE", "QERLKSIVRILEEEERRRRRRFFFFFRRRFFRRFRRFFRFFR"]

        for seq in seqs:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            x = last_hidden_states.detach()
            x = x.mean(axis=1)
            self.assertTrue(torch.all(torch.eq(x, self.model.get_embeddings(seq))))


