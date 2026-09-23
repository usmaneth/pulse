import unittest
from cache_replay import replay


class CacheTest(unittest.TestCase):
    def test_invalid_proposal_and_pinned_limit(self):
        trace = [{'page':p,'bytes':1,'topic':p,'pinned':['a']} for p in ['a','b','c','a']]
        result = replay(trace,2,{'2':{'victim':'a','confidence':1,'snapshot_hash':'invalid'}})
        self.assertEqual(result['hits'],1)
        self.assertEqual(result['decisions'][0]['victim'],'b')
        self.assertFalse(result['decisions'][0]['proposal_accepted'])
        self.assertTrue(result['simulation_only'])


if __name__ == '__main__': unittest.main()
