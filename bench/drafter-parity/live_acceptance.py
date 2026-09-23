#!/usr/bin/env python3
"""Run a bounded paired draft experiment after an exclusive GPU grant."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path('/home/usman')
ARTIFACTS = ROOT/'Documents/spark-orchestration/drafter-artifacts'
BACKEND = ROOT/'llama.cpp-upstream/build-cuda/bin/llama-server'
MODEL = ROOT/'Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf'
DRAFTS = {
    'original': ROOT/'Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf',
    'yarn': ARTIFACTS/'bonsai2-v2-Q4_K_M-yarn32.gguf',
}
PROMPTS = ROOT/'spark-experiments/bonsai2-drafter-2026-09/dflash-training/eval/spec_bench_full/prompts.json'


def request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url+path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def command(variant, port):
    cmd = [str(BACKEND), '-m', str(MODEL), '--host', '127.0.0.1', '--port', str(port),
           '-lv', '4', '-ngl', '999', '-fa', 'on', '-c', '65536', '-np', '1', '-b', '4096', '-ub', '512',
           '--jinja', '--cache-ram', '0', '-ctk', 'f16', '-ctv', 'f16', '--slots']
    if variant != 'k0':
        cmd += ['-md', str(DRAFTS[variant]), '--spec-type', 'draft-dspark', '--spec-draft-n-max', '4', '-ngld', '999']
    return cmd


def make_prompt(url, target_tokens, corpus):
    fixture = json.loads(PROMPTS.read_text())
    case = next(p for p in fixture['prompts'] if p['id'] == 'agent-01')
    system = '\n'.join(fixture['agent_system'])
    result = '\n'.join(case['tool_result'])
    call = case['tool_call']
    def render(size):
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': 'Repository reference material follows.\n'+corpus[:size]+'\nTask:\n'+case['prompt']},
                    {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_fixture', 'type': 'function',
                     'function': {'name': call['name'], 'arguments': json.dumps(call['arguments'])}}]},
                    {'role': 'tool', 'tool_call_id': 'call_fixture', 'content': result}]
        prompt = request(url, '/apply-template', {'messages': messages, 'tools': fixture['agent_tools'], 'add_generation_prompt': True})['prompt']
        tokens = request(url, '/tokenize', {'content': prompt, 'add_special': False})['tokens']
        return prompt, tokens
    low, high = 0, len(corpus)
    while low < high:
        mid = (low+high+1)//2
        if len(render(mid)[1]) <= target_tokens:
            low = mid
        else:
            high = mid-1
    prompt, tokens = render(low)
    if not target_tokens-32 <= len(tokens) <= target_tokens:
        raise ValueError(f'The source corpus does not reach the requested context: {len(tokens)}')
    return {'prompt': prompt, 'tokens': tokens, 'actual_tokens': len(tokens),
            'sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'case': case['id'],
            'task_quality': 'Not automatically assessed. Inspect the output; exactness is a separate result.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=18125)
    parser.add_argument('--contexts', type=int, nargs='+', default=[3072, 8192, 32768])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--exclusive-grant', help='The root-provided grant identifier. This argument cannot acquire a GPU slot.')
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 3 or any(size < 2048 or size > 32768 for size in args.contexts):
        parser.error('Use 1-3 repeats and contexts from 2048 through 32768.')
    schedule = [(size, repeat, variant) for size in args.contexts for repeat in range(args.repeats)
                for variant in (['k0', 'original', 'yarn'] if repeat % 2 == 0 else ['k0', 'yarn', 'original'])]
    if not args.run:
        print(json.dumps({'schedule': schedule, 'commands': {v: command(v, args.port) for v in ['k0', 'original', 'yarn']}, 'GPU_started': False}, indent=2))
        return
    if not args.exclusive_grant:
        raise ValueError('An explicit root GPU grant is required.')
    args.output.mkdir(parents=True, exist_ok=False)
    corpus_files = sorted((ROOT/'llama.cpp-upstream/src').glob('*.cpp'))
    corpus = '\n'.join('\nFILE '+p.name+'\n'+p.read_text() for p in corpus_files)
    (args.output/'corpus.txt').write_text(corpus)
    recorded_backend = json.loads((ROOT/'Documents/spark-orchestration/backend-provenance.json').read_text())
    recorded_variant = json.loads((ARTIFACTS/'bonsai2-v2-Q4_K_M-yarn32.manifest.json').read_text())
    provenance = {'recorded_backend_provenance': recorded_backend,
                  'recorded_variant_manifest': recorded_variant,
                  'hash_scope': 'Hashes originate from the saved manifests; no hash work enters measured request time.',
                  'grant': args.exclusive_grant, 'corpus_sha256': hashlib.sha256(corpus.encode()).hexdigest(),
                  'source_revision': subprocess.check_output(['git', '-C', str(ROOT/'llama.cpp-upstream'), 'rev-parse', 'HEAD'], text=True).strip(),
                  'commands': {v: command(v, args.port) for v in ['k0', 'original', 'yarn']},
                  'seed': 42, 'temperature': 0, 'K': 4, 'n_predict': 256, 'schedule': schedule}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    prompts, baseline = {}, {}
    url = f'http://127.0.0.1:{args.port}'
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    for size, repeat, variant in schedule:
        label = f'{size}-{repeat}-{variant}'
        # Never attach to an existing service or stop a process by name.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', args.port))
        with (args.output/(label+'.log')).open('w') as log:
            process = subprocess.Popen(command(variant, args.port), stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+240
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'The owned server exited: {label}')
                    try:
                        request(url, '/health')
                        break
                    except Exception:
                        if time.monotonic() > deadline:
                            raise TimeoutError('The owned server did not become ready.')
                        time.sleep(.25)
                if variant != 'k0':
                    startup = (args.output/(label+'.log')).read_text()
                    marker = "loading draft model '"+str(DRAFTS[variant])+"'"
                    if marker not in startup:
                        raise ValueError('The log does not identify the expected draft file.')
                    draft_log = startup.split(marker, 1)[1]
                    expected = {'rope scaling': 'yarn' if variant == 'yarn' else 'linear',
                                'freq_scale_train': '0.03125' if variant == 'yarn' else '1',
                                'n_ctx_orig_yarn': '8192' if variant == 'yarn' else '262144',
                                'freq_base_train': '10000000.0',
                                'freq_scale': '0.03125' if variant == 'yarn' else '1'}
                    for name, value in expected.items():
                        if not re.search(re.escape(name)+r'\s*=\s*'+re.escape(value)+r'\s*(?:\n|$)', draft_log):
                            raise ValueError(f'The draft log does not confirm {name}={value}.')
                if size not in prompts:
                    prompts[size] = make_prompt(url, size, corpus)
                    (args.output/f'prompt-{size}.json').write_text(json.dumps(prompts[size], indent=2))
                fixture = prompts[size]
                started = time.perf_counter()
                response = request(url, '/completion', {'prompt': fixture['tokens'], 'n_predict': 256,
                    'temperature': 0, 'seed': 42, 'cache_prompt': False, 'return_tokens': True})
                elapsed = time.perf_counter()-started
                (args.output/(label+'.response.json')).write_text(json.dumps({'elapsed_s': elapsed, 'response': response}, indent=2))
                timings = response['timings']
                if variant != 'k0' and not all(k in timings for k in ['draft_n', 'draft_n_accepted']):
                    raise ValueError('The server omitted the draft counters.')
                if timings.get('cache_n', 0) != 0:
                    raise ValueError('A cold request reused cached tokens.')
                if abs(timings['prompt_n']-fixture['actual_tokens']) > 1:
                    raise ValueError('The server prompt count differs from the fixture.')
                content = response.get('content', '')
                output_tokens = response.get('tokens')
                if not isinstance(output_tokens, list) or not output_tokens or not all(isinstance(t, int) for t in output_tokens):
                    raise ValueError('The server did not return output token IDs.')
                if variant == 'k0':
                    baseline[size, repeat] = (content, output_tokens)
                row = {'context': size, 'repeat': repeat, 'variant': variant, 'elapsed_s': elapsed,
                       'output_matches_k0': (content, output_tokens) == baseline[size, repeat],
                       'text_matches_k0': content == baseline[size, repeat][0],
                       'tokens_match_k0': output_tokens == baseline[size, repeat][1],
                       'output_limit': 256, 'stopped_at_output_limit': response.get('stop_type') == 'limit' or response.get('stopped_limit', False),
                       'comparison_scope': 'At most 256 output tokens; a limit stop is not completed task success.',
                       'task_quality': fixture['task_quality'], 'prompt_sha256': fixture['sha256'],
                       'acceptance': timings['draft_n_accepted']/timings['draft_n'] if timings.get('draft_n') else None,
                       'response': response}
                with (args.output/'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(json.dumps({k: v for k, v in row.items() if k != 'response'}), flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == '__main__':
    main()
