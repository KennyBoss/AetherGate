from context import legacy
from completion import edits,beam_select,source_windows,complete
import unittest,numpy as np
class CompletionTests(unittest.TestCase):
 def test_structural_neighbors_and_layout_are_exhausted_without_new_opcodes(self):
  row={'code':[3,7,4]+[0]*9,'layout':[0,1]};rows=list(edits(row));keys={(tuple(r['code']),tuple(r['layout'])) for r in rows};self.assertEqual(len(keys),len(rows));self.assertIn((tuple([7,4]+[0]*10),(0,1)),keys);self.assertIn((tuple([0,3,7,4]+[0]*8),(0,1)),keys);self.assertIn((tuple(row['code']),(0,2)),keys)
  for r in rows:self.assertEqual(len(r['code']),12);self.assertTrue(set(r['code'])<=set(legacy.ALPHABET));self.assertNotIn(35,r['code'])
 def test_neutral_syntax_variants_survive_beam(self):
  rows=[{'code':[i],'layout':[0,1]} for i in range(20)];v=np.ones((20,9));o=np.zeros((20,25,2));picked=beam_select(rows,v,o,'order');self.assertEqual(len(picked),16);self.assertGreater(len({tuple(r['code']) for r in picked}),1)
 def test_unconditional_windows_are_not_filtered_out(self):
  rows=source_windows([{'archive':[[3,7,4,9]]}]);self.assertTrue(any(r['code']==[3,7,4] for r in rows));self.assertTrue(all(r['outputs']==[3,4] for r in rows))
 def test_complete_inventory_spends_no_budget(self):
  r=complete({}, {'incumbent_blocks':[{'behavior':'swap'},{'behavior':'order_ascending'}],'blocks':[]},0,100);self.assertEqual(r['missing_before'],[]);self.assertEqual(r['preparation_case_steps_used'],0);self.assertEqual(r['status'],'complete')
if __name__=='__main__':unittest.main()
