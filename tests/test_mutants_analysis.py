import unittest
import numpy as np
import pandas as pd
from src.scripts.mutants_analysis import get_differences, plot_mutants_graph


class Test(unittest.TestCase):

    def test_mutants_analysis_1(self):
        """
        Testing the difference between "['Tyrosine-protein kinase BTK', 'Tyrosine-protein kinase BTK [C481S]']"
        """
        df = pd.read_csv('../data/mutants.csv')
        row = df.iloc[46]
        differences = get_differences(row['WT Target Name'], row['Target Names'],
                                      row['BindingDB Target Chain Sequence'])
        self.assertEqual(differences.shape[0], 1)
        self.assertEqual(differences['Positions'][0], np.array([480]))
        self.assertEqual(differences['Alignment Reference'][0][480], 'C')
        self.assertEqual(differences['Alignment Mutant'][0][480], 'S')

    def test_mutants_analysis_2(self):
        """
        Testing the difference between "['Beta-secretase 1', 'Beta-secretase 1 [1-460]']"
        """
        df = pd.read_csv('../data/mutants.csv')
        row = df.iloc[145]
        differences = get_differences(row['WT Target Name'], row['Target Names'],
                                      row['BindingDB Target Chain Sequence'])
        self.assertEqual(differences.shape[0], 1)
        self.assertTrue(np.array_equal(differences['Positions'][0], np.array(range(460, 501))))
        self.assertEqual(differences['Alignment Reference'][0][500], 'K')
        self.assertEqual(differences['Alignment Mutant'][0][460:501], '-'*41)

    def test_mutants_analysis_3(self):
        """
        Testing the difference between "['Coagulation factor XIII A chain', 'Coagulation factor XIII A chain [Q652E]']"
        """
        df = pd.read_csv('../data/mutants.csv')
        row = df.iloc[161]
        differences = get_differences(row['WT Target Name'], row['Target Names'],
                                      row['BindingDB Target Chain Sequence'])
        self.assertEqual(differences.shape[0], 1)
        self.assertTrue(np.array_equal(differences['Positions'][0], np.array([651])))
        self.assertEqual(differences['Alignment Reference'][0][651], 'Q')
        self.assertEqual(differences['Alignment Mutant'][0][651], 'E')

    def test_mutants_analysis_4(self):
        """
        Testing the difference between "['Proto-oncogene tyrosine-protein kinase ROS',
        'Proto-oncogene tyrosine-protein kinase ROS [D2033N]', 'Proto-oncogene tyrosine-protein kinase ROS [G2032R]',
        'Proto-oncogene tyrosine-protein kinase ROS [L2026M]']"
        """
        df = pd.read_csv('../data/mutants.csv')
        row = df.iloc[189]
        differences = get_differences(row['WT Target Name'], row['Target Names'],
                                      row['BindingDB Target Chain Sequence'])
        self.assertEqual(differences.shape[0], 3)
        # Check first pair
        self.assertTrue(np.array_equal(differences['Positions'][0], np.array([2032])))
        self.assertEqual(differences['Alignment Reference'][0][2032], 'D')
        self.assertEqual(differences['Alignment Mutant'][0][2032], 'N')

        # Check second pair
        self.assertTrue(np.array_equal(differences['Positions'][1], np.array([2031])))
        self.assertEqual(differences['Alignment Reference'][1][2031], 'G')
        self.assertEqual(differences['Alignment Mutant'][1][2031], 'R')

        # Check third pair
        self.assertTrue(np.array_equal(differences['Positions'][2], np.array([2025])))
        self.assertEqual(differences['Alignment Reference'][2][2025], 'L')
        self.assertEqual(differences['Alignment Mutant'][2][2025], 'M')

    def test_mutants_analysis_5(self):
        """
        Testing the difference between "['Proto-oncogene tyrosine-protein kinase receptor Ret',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114,G810R]',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114,V804M]',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [658-1114]',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [G810R]',
        'Proto-oncogene tyrosine-protein kinase receptor Ret [V804M]']"
        """
        df = pd.read_csv('../data/mutants.csv')
        row = df.iloc[267]
        differences = get_differences(row['WT Target Name'], row['Target Names'],
                                      row['BindingDB Target Chain Sequence'])
        # Check the number of pairs in the dataframe
        self.assertEqual(differences.shape[0], 5)

        # Check first pair
        self.assertTrue(np.array_equal(differences['Positions'][0], np.append(np.arange(657), 809)))
        self.assertEqual(differences['Alignment Reference'][0][809], 'G')
        self.assertEqual(differences['Alignment Mutant'][0][809], 'R')
        self.assertEqual(differences['Alignment Mutant'][0][0:657], '-'*657)

        # Check second pair
        self.assertTrue(np.array_equal(differences['Positions'][1], np.append(np.arange(657), 803)))
        self.assertEqual(differences['Alignment Reference'][1][803], 'V')
        self.assertEqual(differences['Alignment Mutant'][1][803], 'M')
        self.assertEqual(differences['Alignment Mutant'][1][0:657], '-'*657)

        # # Check third pair
        self.assertTrue(np.array_equal(differences['Positions'][2], np.array(range(0, 657))))
        self.assertEqual(differences['Alignment Mutant'][2][0:657], '-'*657)

        # Check fourth pair
        self.assertTrue(np.array_equal(differences['Positions'][3], np.array([809])))
        self.assertEqual(differences['Alignment Reference'][3][809], 'G')
        self.assertEqual(differences['Alignment Mutant'][3][809], 'R')

        # Check fifth pair
        self.assertTrue(np.array_equal(differences['Positions'][4], np.array([803])))
        self.assertEqual(differences['Alignment Reference'][4][803], 'V')
        self.assertEqual(differences['Alignment Mutant'][4][803], 'M')

    def test_mutants_analysis_6(self):
        """
        Testing the difference between "['RAC-alpha serine/threonine-protein kinase', 'RAC-alpha serine/threonine-protein kinase [139-480,S378A,S381A,T450D,S473D]']"
        """
        df = pd.read_csv('../data/mutants.csv')
        row = df.iloc[0]
        differences = get_differences(row['WT Target Name'], row['Target Names'], row['BindingDB Target Chain Sequence'])
        self.assertEqual(differences.shape[0], 1)
        self.assertTrue(np.array_equal(differences['Positions'][0], np.append(np.arange(138), [377, 380, 449, 472])))
        # Position 377
        self.assertEqual(differences['Alignment Reference'][0][377], 'S')
        self.assertEqual(differences['Alignment Mutant'][0][377], 'A')

        # Position 380
        self.assertEqual(differences['Alignment Reference'][0][380], 'S')
        self.assertEqual(differences['Alignment Mutant'][0][380], 'A')

        # Position 449
        self.assertEqual(differences['Alignment Reference'][0][449], 'T')
        self.assertEqual(differences['Alignment Mutant'][0][449], 'D')

        # Position 472
        self.assertEqual(differences['Alignment Reference'][0][472], 'S')
        self.assertEqual(differences['Alignment Mutant'][0][472], 'D')

    def test_graph_mutants(self):
        df_mutants = pd.read_csv('../data/mutants.csv')
        df_merged = pd.read_csv('../data/merged_df.csv')
        row = df_mutants.iloc[267]
        print(row)
        plot_mutants_graph(row, df_merged)
