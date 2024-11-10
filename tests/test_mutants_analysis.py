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
        self.assertEqual(differences['Alignment 2'][0][460:501], '-'*41)

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

    def test_mutants_analysis_4(self):
        """
        Testing the difference between "['Proto-oncogene tyrosine-protein kinase ROS',
        'Proto-oncogene tyrosine-protein kinase ROS [D2033N]', 'Proto-oncogene tyrosine-protein kinase ROS [G2032R]',
        'Proto-oncogene tyrosine-protein kinase ROS [L2026M]']"
        """
        file_path = '../data/merged_df.csv'
        differences = get_differences("['Proto-oncogene tyrosine-protein kinase ROS', 'Proto-oncogene "
                                      "tyrosine-protein kinase ROS [D2033N]', 'Proto-oncogene tyrosine-protein kinase "
                                      "ROS [G2032R]', 'Proto-oncogene tyrosine-protein kinase ROS [L2026M]']",
                                      file_path)
        self.assertEqual(differences.shape[0], 3)
        # Check first pair
        self.assertTrue(np.array_equal(differences['Positions'][0], np.array([2032])))
        self.assertEqual(differences['Alignment 1'][0][2032], 'D')
        self.assertEqual(differences['Alignment 2'][0][2032], 'N')

        # Check second pair
        self.assertTrue(np.array_equal(differences['Positions'][1], np.array([2031])))
        self.assertEqual(differences['Alignment 1'][1][2031], 'G')
        self.assertEqual(differences['Alignment 2'][1][2031], 'R')

        # Check third pair
        self.assertTrue(np.array_equal(differences['Positions'][2], np.array([2025])))
        self.assertEqual(differences['Alignment 1'][2][2025], 'L')
        self.assertEqual(differences['Alignment 2'][2][2025], 'M')

    def test_mutants_analysis_5(self):
        """
        Testing the difference between "['Proto-oncogene tyrosine-protein kinase receptor Ret',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114,G810R]',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114,V804M]',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114]',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [G810R]',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [V804M]']"
        """
        file_path = '../data/merged_df.csv'
        differences = get_differences("['Proto-oncogene tyrosine-protein kinase receptor Ret', "
                                      "'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114,G810R]', "
                                      "'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114,V804M]', "
                                      "'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114]', "
                                      "'Proto-oncogene tyrosine-protein kinase receptor Ret [G810R]', "
                                      "'Proto-oncogene tyrosine-protein kinase receptor Ret [V804M]']",
                                      file_path)
        # Check the number of pairs in the dataframe
        self.assertEqual(differences.shape[0], 5)

        # Check first pair
        self.assertTrue(np.array_equal(differences['Positions'][0], np.append(np.arange(657), 809)))
        self.assertEqual(differences['Alignment 1'][0][809], 'G')
        self.assertEqual(differences['Alignment 2'][0][809], 'R')
        self.assertEqual(differences['Alignment 2'][0][0:657], '-'*657)

        # Check second pair
        self.assertTrue(np.array_equal(differences['Positions'][1], np.append(np.arange(657), 803)))
        self.assertEqual(differences['Alignment 1'][1][803], 'V')
        self.assertEqual(differences['Alignment 2'][1][803], 'M')
        self.assertEqual(differences['Alignment 2'][1][0:657], '-'*657)

        # # Check third pair
        self.assertTrue(np.array_equal(differences['Positions'][2], np.array(range(0, 657))))
        self.assertEqual(differences['Alignment 2'][2][0:657], '-'*657)

        # Check fourth pair
        self.assertTrue(np.array_equal(differences['Positions'][3], np.array([809])))
        self.assertEqual(differences['Alignment 1'][3][809], 'G')
        self.assertEqual(differences['Alignment 2'][3][809], 'R')

        # Check fifth pair
        self.assertTrue(np.array_equal(differences['Positions'][4], np.array([803])))
        self.assertEqual(differences['Alignment 1'][4][803], 'V')
        self.assertEqual(differences['Alignment 2'][4][803], 'M')