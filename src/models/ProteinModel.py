import torch
from transformers import pipeline, AutoTokenizer


class ProteinModel:
    """
    ProteinModel class using ESM2.
    """

    def __init__(self, model='facebook/esm2_t6_8M_UR50D'):
        """
        Constructor method.
        :param model: ESM model.
        """

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Set the model
        self.pipe = pipeline("fill-mask", model=model, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def get_probabilities_unmasked(self, protein_sequence, position):
        """
        Returns probability of having any amino acid at the specific position in the protein sequence.
        :param protein_sequence: protein sequence
        :param position: position of the amino acid in the protein sequence that wants to be masked
        :return probability of having any amino acid at the specific position in the protein sequence
        """
        masked_sequence = protein_sequence[:position] + "<mask>" + protein_sequence[position + 1:]
        return self.get_probabilities_masked(masked_sequence)

    def get_probabilities_masked(self, masked_sequence):
        """
        Returns probability of having any amino acid at the masked position in the protein sequence.
        :param masked_sequence: protein sequence with a masked amino acid
        :return probability of having any amino acid at the masked position in the protein sequence
        """
        with torch.no_grad():
            predictions = self.pipe(masked_sequence, top_k=33)
            return predictions

    def get_embeddings(self, sequence):
        """
        Returns the embeddings for a protein sequence.
        Code taken from: https://github.com/facebookresearch/esm/issues/348
        :param sequence: protein sequence to be embedded
        :return embeddings: embeddings for a protein sequence
        """
        with torch.no_grad():
            inputs = self.tokenizer(sequence, return_tensors="pt")
            outputs = self.pipe.model.esm(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.last_hidden_state
            embeddings = last_hidden_states.detach()
            return embeddings.mean(axis=1)
