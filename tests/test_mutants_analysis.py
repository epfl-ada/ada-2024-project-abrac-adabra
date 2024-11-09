import unittest
import numpy as np
from src.scripts.mutants_analysis import get_differences


class Test(unittest.TestCase):

    def test_mutants_analysis_1(self):
        """
        Testing the difference between "['Tyrosine-protein kinase BTK', 'Tyrosine-protein kinase BTK [C481S]']"
        """
        file_path = '../data/merged_df.csv'
        differences = get_differences("['Tyrosine-protein kinase BTK', 'Tyrosine-protein kinase BTK [C481S]']", file_path)
        self.assertEqual(differences.shape[0], 1)
        self.assertEqual(differences['Positions'][0], np.array([480]))
        self.assertEqual(differences['Alignment 1'][0][480], 'C')
        self.assertEqual(differences['Alignment 2'][0][480], 'S')

    def test_mutants_analysis_2(self):
        """
        Testing the difference between "['Beta-secretase 1', 'Beta-secretase 1 [1-460]']"
        """
        file_path = '../data/merged_df.csv'
        differences = get_differences("['Beta-secretase 1', 'Beta-secretase 1 [1-460]']", file_path)
        self.assertEqual(differences.shape[0], 1)
        self.assertTrue(np.array_equal(differences['Positions'][0], np.array(range(460, 501))))
        self.assertEqual(differences['Alignment 1'][0][500], 'K')
        self.assertEqual(differences['Alignment 2'][0][500], '-')

    def test_mutants_analysis_3(self):
        """
        Testing the difference between "['Coagulation factor XIII A chain', 'Coagulation factor XIII A chain [Q652E]']"
        """
        file_path = '../data/merged_df.csv'
        differences = get_differences("['Coagulation factor XIII A chain', 'Coagulation factor XIII A chain [Q652E]']",
                                      file_path)
        self.assertEqual(differences.shape[0], 1)
        self.assertTrue(np.array_equal(differences['Positions'][0], np.array([651])))
        self.assertEqual(differences['Alignment 1'][0][651], 'Q')
        self.assertEqual(differences['Alignment 2'][0][651], 'E')


