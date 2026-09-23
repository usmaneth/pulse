#!/usr/bin/env python3
"""Request bounded policy proposals. Keep enforcement local."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

MODEL = 'jev-1.13.0'
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
CHOICES = {
    'speculation': {'k0': 'Disable speculative proposals.', 'k2': 'Propose at most two draft tokens.',
                    'k4': 'Propose at most four draft tokens.', 'k7': 'Propose at most seven draft tokens.',
                    'abstain': 'Insufficient evidence. Retain the local default.'},
    'kernel': {'reference': 'Use the validated current backend kernels.',
               'graph': 'Use the complete graph implementation only if the local manifest permits it.',
               'abstain': 'Insufficient evidence. Retain the current backend.'},
    'cache': {'lru': 'Evict the oldest eligible prefix.',
              'semantic': 'Use semantic relevance only in the separate cache replay simulation.',
              'abstain': 'Insufficient evidence. Retain LRU.'},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bucket_telemetry(measurements):
    """Calculate telemetry categories locally. Do not ask Jev to do arithmetic."""
    result = {}
    for candidate, measurement in measurements.items():
        if candidate not in CHOICES['speculation'] or candidate == 'abstain':
            raise ValueError('Unknown measurement candidate.')
        elapsed = measurement['wall_s']
        tokens = measurement['output_tokens']
        if not math.isfinite(elapsed) or elapsed <= 0 or type(tokens) is not int or tokens < 0:
            raise ValueError('Invalid measurement.')
        rate = tokens/elapsed
        acceptance = measurement.get('accepted', 0)/max(1, measurement.get('drafted', 0))
        result[candidate] = {'net_rate_bucket': 'fast' if rate >= 25 else 'moderate' if rate >= 10 else 'slow',
                             'acceptance_bucket': 'high' if acceptance >= .7 else 'moderate' if acceptance >= .4 else 'low',
                             'quality_gate': 'passed' if measurement.get('quality_passed') is True else 'unproved',
                             'source': measurement['source']}
    return result


def request_body(state, constraints):
    # Only this explicit public projection reaches the service. Never include answers or private source text.
    public = {key: state[key] for key in ['workload_description', 'context_bucket', 'telemetry_buckets']}
    public['legal_candidates'] = constraints['legal']
    return {'model': MODEL, 'state': public, 'questions': {
        name: {'type': 'choice', 'instructions':
               f'Choose the {name} policy for the NEXT request from the legal candidates. '
               'Use the semantic workload and supplied categories. Do not calculate rates. '
               'Abstain when evidence is insufficient. A quality-unproved measurement does not prove safety or correctness.',
               'criteria': choices} for name, choices in CHOICES.items()}}


def validate_answer(name, answer):
    if answer.get('type') != 'choice' or answer.get('choice') not in CHOICES[name]:
        raise ValueError('The response contains an unknown choice.')
    probabilities = answer.get('probabilities', {})
    if set(probabilities) != set(CHOICES[name]):
        raise ValueError('The response distribution differs from the requested choices.')
    values = list(probabilities.values())+[answer.get('confidence')]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('The response contains an invalid probability or confidence.')
    if abs(sum(probabilities.values())-1) > .02:
        raise ValueError('The response probabilities do not sum to one.')
    if probabilities[answer['choice']] < max(probabilities.values()):
        raise ValueError('The selected choice does not have maximum probability.')


def validate_constraints(constraints):
    default = {'speculation': 'k4', 'kernel': 'reference', 'cache': 'lru'}
    for field in ['required_bytes', 'available_bytes', 'server_max_k', 'block_cap']:
        if type(constraints[field]) is not int or constraints[field] < 0:
            raise ValueError('Local capacity values must be nonnegative integers.')
    if set(constraints['legal']) != set(CHOICES) or any(set(v)-set(CHOICES[k]) for k,v in constraints['legal'].items()):
        raise ValueError('The legal candidate manifest contains unknown choices.')
    if constraints['server_max_k'] != 7 or constraints['block_cap'] != 7:
        raise ValueError('This experiment requires server and draft caps of seven.')
    if any(default[k] not in constraints['legal'][k] for k in default):
        raise ValueError('The local fallback must be legal.')
    if not constraints['target_verification_required']:
        raise ValueError('Numerical target verification cannot be disabled.')
    return default if constraints['required_bytes'] <= constraints['available_bytes'] else None


def local_fallback(constraints, reason):
    selected = validate_constraints(constraints)
    return {'selected': selected, 'abstained': True, 'source': 'local_policy',
            'raw': None, 'confidence': None,
            'reason': reason if selected is not None else 'Local memory admission failed.'}


def decide(response, state, constraints, requested_hash, created_at, now=None, mode="advice"):
    """Validate a proposal against current local constraints and the request snapshot."""
    if mode not in ('advice', 'experiment'):
        raise ValueError('Unknown policy mode.')
    now = time.time() if now is None else now
    default = validate_constraints(constraints)
    if default is None:
        return local_fallback(constraints, 'Local memory admission failed.')
    if requested_hash != digest({'state': state, 'constraints': constraints}) or now-created_at > 30 or now < created_at:
        return {'selected': default if mode == 'advice' else None, 'abstained': True, 'reason': 'The proposal is stale or the snapshot changed.', 'mode': mode}
    if response.get('model') != MODEL:
        raise ValueError('The service returned an unexpected model.')
    if mode == 'experiment':
        rankings = {}
        for name in CHOICES:
            answer = response['answers'][name]
            validate_answer(name, answer)
            legal = [candidate for candidate in constraints['legal'][name] if candidate != 'abstain']
            if name == 'speculation':
                legal = [c for c in legal if int(c[1:]) <= min(constraints['server_max_k'], constraints['block_cap'])]
            if name == 'cache' and not constraints['simulation_only']:
                legal = [c for c in legal if c != 'semantic']
            rankings[name] = {'trial_order': sorted(set(legal), key=lambda c: (-answer['probabilities'][c], c)),
                              'raw': answer, 'confidence_gate_applied': False}
        return {'mode': 'experiment', 'status': 'candidate proposal only', 'rankings': rankings,
                'selected': None, 'auto_promoted': False, 'request_settings': None,
                'promotion_requires': ['numerical correctness', 'task tests', 'local resource admission', 'measured paired benefit'],
                'application': 'Trial order only. Evaluate every legal candidate. No production policy change.'}
    selected, decisions = {}, {}
    for name in CHOICES:
        answer = response['answers'][name]
        validate_answer(name, answer)
        choice = answer['choice']
        eligible = choice in constraints['legal'][name]
        if name == 'speculation' and choice != 'abstain':
            eligible = eligible and int(choice[1:]) <= min(constraints['server_max_k'], constraints['block_cap'])
        if name == 'kernel' and choice == 'graph':
            eligible = eligible and constraints.get('graph_correctness_passed') is True
        if name == 'cache' and choice == 'semantic':
            eligible = eligible and constraints['simulation_only'] is True
        accept = eligible and choice != 'abstain' and answer['confidence'] >= .8
        selected[name] = choice if accept else default[name]
        decisions[name] = {'raw': answer, 'accepted': accept,
                           'reason': 'accepted proposal' if accept else 'abstention, low confidence, or local constraint'}
    return {'selected': selected, 'decisions': decisions, 'abstained': any(not d['accepted'] for d in decisions.values()),
            'threshold': .8, 'threshold_status': 'Uncalibrated experiment threshold.',
            'request_settings': {'speculative.n_max': int(selected['speculation'][1:])},
            'application': 'Next request boundary only. No active request or allocator mutation.'}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('The API redirect was rejected.')


def api_call(body, secret_file):
    # Parse the environment file as data. Never execute it or log its contents.
    values = {}
    for line in Path(secret_file).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.removeprefix('export ').split('=', 1)
            values[key.strip()] = value.strip().strip('"\'')
    token = next((values[k] for k in ['TYPESAFE_API_KEY', 'JEV_API_KEY'] if values.get(k)), None)
    if not token:
        raise ValueError('The secure file lacks the expected API key variable.')
    req = urllib.request.Request(ENDPOINT, json.dumps(body).encode(),
                                 {'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'})
    start = time.perf_counter()
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(req, timeout=10) as response:
        raw = response.read(1024*1024+1)
        if len(raw) > 1024*1024:
            raise ValueError('The API response exceeds the size limit.')
        result = json.loads(raw)
    return result, (time.perf_counter()-start)*1000


class NextRequestProposal:
    """Start the API call outside the GPU path. Never wait at the request boundary."""
    def __init__(self, state, constraints, secret_file):
        self.created_at = time.time()
        self.snapshot_hash = digest({'state': state, 'constraints': constraints})
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(api_call, request_body(state, constraints), secret_file)

    def poll(self, state, constraints):
        if not self.future.done():
            return local_fallback(constraints, 'pending')
        try:
            try:
                response, elapsed = self.future.result()
            except Exception:
                return local_fallback(constraints, 'api_error')
            try:
                result = decide(response, state, constraints, self.snapshot_hash, self.created_at)
            except (KeyError, TypeError, ValueError):
                return local_fallback(constraints, 'invalid_response')
            result['api_elapsed_ms'] = elapsed
            return result
        finally:
            self.executor.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--mode', choices=['advice', 'experiment'], default='advice')
    parser.add_argument('--secret-file', type=Path, default=Path('/home/usman/.config/pulse/typesafe.env'))
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    state, constraints = fixture['state'], fixture['constraints']
    body = request_body(state, constraints)
    record = {'request': body, 'fixture_sha256': digest(fixture), 'mode': 'proposal only'}
    if args.live:
        created = time.time()
        response, elapsed = api_call(body, args.secret_file)
        record.update({'response': response, 'elapsed_ms': elapsed,
                       'decision': decide(response, state, constraints, digest(fixture), created, mode=args.mode)})
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'mode': record['mode'], 'elapsed_ms': record.get('elapsed_ms'), 'decision': record.get('decision')}))


if __name__ == '__main__':
    main()
