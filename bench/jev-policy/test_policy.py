import copy
import time
import unittest
from policy import CHOICES, MODEL, decide, digest, request_body


def fixture():
    return {'state': {'workload_description': 'Inspect source and propose a patch.', 'context_bucket': 'short', 'telemetry_buckets': {}},
            'constraints': {'server_max_k': 7, 'block_cap': 7, 'legal': {'speculation': ['k0','k2','k4','k7'],
                'kernel': ['reference'], 'cache': ['lru']}, 'required_bytes': 10, 'available_bytes': 20,
                'target_verification_required': True, 'graph_correctness_passed': False, 'simulation_only': False}}


def response(choices=None):
    choices = choices or {'speculation': 'k7', 'kernel': 'graph', 'cache': 'semantic'}
    return {'model': MODEL, 'answers': {name: {'type': 'choice', 'choice': choices[name], 'confidence': 1.,
        'probabilities': {key: float(key == choices[name]) for key in options}} for name, options in CHOICES.items()}}


class PolicyTests(unittest.TestCase):
    def run_decision(self, f, r=None, **kwargs):
        return decide(r or response(), f['state'], f['constraints'], digest(f), time.time(), **kwargs)

    def test_legal_spec_and_illegal_kernel_cache(self):
        result = self.run_decision(fixture())
        self.assertEqual(result['selected'], {'speculation':'k7','kernel':'reference','cache':'lru'})
        self.assertTrue(result['abstained'])

    def test_memory_and_verification_enforcement(self):
        f = fixture(); f['constraints']['available_bytes'] = 9
        self.assertIsNone(self.run_decision(f)['selected'])
        f = fixture(); f['constraints']['target_verification_required'] = False
        with self.assertRaises(ValueError): self.run_decision(f)

    def test_bad_distribution_and_low_confidence(self):
        r = response(); r['answers']['speculation']['confidence'] = float('nan')
        with self.assertRaises(ValueError): self.run_decision(fixture(), r)
        r = response(); r['answers']['speculation']['confidence'] = .4
        self.assertEqual(self.run_decision(fixture(), r)['selected']['speculation'], 'k4')

    def test_stale_snapshot_falls_back(self):
        f = fixture()
        result = decide(response(),f['state'],f['constraints'],digest(f),time.time()-60)
        self.assertEqual(result['selected']['speculation'],'k4')

    def test_private_answers_excluded(self):
        f = fixture(); f['state']['expected_answer'] = 'PRIVATE'
        body = request_body(f['state'], f['constraints'])
        self.assertNotIn('PRIVATE',str(body))



class ExperimentTests(unittest.TestCase):
    def test_low_confidence_ranks_all_legal_candidates(self):
        f = fixture(); r = response()
        r['answers']['speculation']['confidence'] = .22
        result = decide(r,f['state'],f['constraints'],digest(f),time.time(),mode='experiment')
        self.assertEqual(set(result['rankings']['speculation']['trial_order']),{'k0','k2','k4','k7'})
        self.assertEqual(result['rankings']['kernel']['trial_order'],['reference'])
        self.assertIsNone(result['selected'])
        self.assertFalse(result['auto_promoted'])

    def test_invalid_candidate_cannot_enter_ranking(self):
        f = fixture(); f['constraints']['legal']['speculation'].append('k99')
        with self.assertRaises(ValueError):
            decide(response(),f['state'],f['constraints'],digest(f),time.time(),mode='experiment')
        f = fixture(); r = response(); r['answers']['speculation']['probabilities']['k99'] = .1
        with self.assertRaises(ValueError):
            decide(r,f['state'],f['constraints'],digest(f),time.time(),mode='experiment')

    def test_experiment_stale_and_memory_fail_closed(self):
        f = fixture()
        self.assertIsNone(decide(response(),f['state'],f['constraints'],digest(f),time.time()-60,mode='experiment')['selected'])
        f['constraints']['available_bytes'] = 0
        result = decide(response(),f['state'],f['constraints'],digest(f),time.time(),mode='experiment')
        self.assertNotIn('rankings',result)


class AsyncSafetyTests(unittest.TestCase):
    def test_pending_and_error_are_local_without_confidence(self):
        from concurrent.futures import Future
        from unittest.mock import Mock
        from policy import NextRequestProposal
        f = fixture()
        proposal = NextRequestProposal.__new__(NextRequestProposal)
        proposal.future = Future(); proposal.executor = Mock()
        for reason in ['pending','api_error']:
            if reason == 'api_error': proposal.future.set_exception(RuntimeError('private detail'))
            result = proposal.poll(f['state'],f['constraints'])
            self.assertEqual(result['source'],'local_policy')
            self.assertIsNone(result['raw']); self.assertIsNone(result['confidence'])
            self.assertEqual(result['reason'],reason)
            self.assertNotIn('private detail',str(result))
        f['constraints']['available_bytes'] = 0
        self.assertIsNone(proposal.poll(f['state'],f['constraints'])['selected'])

    def test_redirect_is_rejected_before_forwarding(self):
        import urllib.request
        from policy import NoRedirect
        req = urllib.request.Request('https://api.typesafe.ai/v1/systemone',headers={'Authorization':'Bearer synthetic'})
        with self.assertRaisesRegex(ValueError,'redirect was rejected'):
            NoRedirect().redirect_request(req,None,302,'Found',{},'https://other.invalid/')

if __name__ == '__main__': unittest.main()
