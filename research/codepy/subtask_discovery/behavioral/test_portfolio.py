import unittest
from portfolio import budget_ledger,choose_training_program
class PortfolioTests(unittest.TestCase):
 def test_total_budget_pays_for_both_libraries(self):
  for n in [0,1000000,4718592]:
   q=budget_ledger(237109248,4718592,900000,n);self.assertTrue(q['within_original_total_budget']);self.assertLessEqual(q['new_total_units'],q['baseline_total_units'])
  self.assertFalse(budget_ledger(0,0,0,4718593)['within_original_total_budget'])
 def test_selection_uses_training_not_certificate_and_stable_ties(self):
  q={'arguments':{'training_objective':1.,'certificate':{'passed':True}},'behavioral':{'training_objective':0.,'certificate':{'passed':False}}};self.assertEqual(choose_training_program(q),'behavioral')
  q['arguments']['training_objective']=0.;self.assertEqual(choose_training_program(q),'arguments')
if __name__=='__main__':unittest.main()
