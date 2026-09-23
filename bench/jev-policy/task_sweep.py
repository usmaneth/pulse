#!/usr/bin/env python3
"""Evaluate completed tasks across bounded draft depths after an exclusive grant."""
import argparse
import contextlib
import hashlib
import json
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'agent-eval'))
import runner as evaluation
import code_fixture as code_validator
from code_fixture import validate as validate_code
sys.path.insert(0,str(HERE.parent/'drafter-parity'))
import live_acceptance as backend
from policy import digest

TOOLS = [{'type':'function','function':{'name':'multiply','description':'Multiply two integers.',
         'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},
                       'required':['a','b'],'additionalProperties':False}}}]


def tool_result(first):
    calls = first['tool_calls']
    if first['finish_reason'] != 'tool_calls' or len(calls) != 1:
        raise ValueError('The task did not return one complete tool call.')
    call = calls[0]
    arguments = json.loads(call['function']['arguments'])
    if not call['id'] or call['function']['name'] != 'multiply' or arguments != {'a':37,'b':19}:
        raise ValueError('The tool name or arguments differ from the task contract.')
    if any(type(v) is not int for v in arguments.values()):
        raise ValueError('The tool arguments must be integers.')
    return {'role':'tool','tool_call_id':call['id'],'content':json.dumps({'result':arguments['a']*arguments['b']})}


def config(url, k, reasoning_budget=None):
    options={"speculative.n_max":k}
    if reasoning_budget is not None: options["reasoning_budget_tokens"]=reasoning_budget
    return {'url':url+'/v1','model':'bonsai','max_tokens':2048,'timeout_s':900,
            'request_options':options,
            'token_counter':[sys.executable,str(HERE.parent/'agent-eval/llama_hooks.py'),'count','--url',url]}


def fixture(cfg, task, size, corpus):
    tools = TOOLS if task['id'] == 'tool-multiply' else []
    system = ('Use the multiply tool. After its result, reply with only the integer answer.' if tools else
              'Return only the requested Python function. Ignore unrelated reference source.')
    def make(n):
        messages = [{'role':'system','content':system},{'role':'user','content':'Reference source:\n'+corpus[:n]+'\nTask:\n'+task['task']}]
        return messages, evaluation.count_tokens(cfg,messages,tools)
    low, high = 0,len(corpus)
    while low < high:
        mid = (low+high+1)//2
        if make(mid)[1] <= size: low = mid
        else: high = mid-1
    messages,count = make(low)
    if not size-32 <= count <= size:
        raise ValueError('The fixture cannot reach the requested context.')
    return {'messages':messages,'tools':tools,'prompt_tokens':count,'sha256':digest(messages)}


@contextlib.contextmanager
def server(port, directory, no_draft=False):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',port))
    command = backend.command('k0' if no_draft else 'yarn',port)
    command[command.index('-lv')+1] = '5' # The log confirms the actual per-request draft cap.
    slot_path=directory/'slot-state'
    slot_path.mkdir(exist_ok=True)
    command += ['--reasoning-format','deepseek','--reasoning-preserve','--slot-save-path',str(slot_path)]
    if not no_draft: command[command.index('--spec-draft-n-max')+1] = '7'
    log_path = directory/('no-draft.log' if no_draft else 'draft-max7.log')
    with log_path.open('w') as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+240
            url = f'http://127.0.0.1:{port}'
            while True:
                if child.poll() is not None: raise RuntimeError('The owned backend exited before readiness.')
                try:
                    backend.request(url,'/health'); break
                except Exception:
                    if time.monotonic() >= deadline: raise TimeoutError('The owned backend did not become ready.')
                    time.sleep(.25)
            if not no_draft:
                text = log_path.read_text()
                marker = "loading draft model '"+str(backend.DRAFTS['yarn'])+"'"
                if marker not in text: raise ValueError('The expected corrected draft did not load.')
                text = text.split(marker,1)[1]
                for pattern in [r'rope scaling\s*=\s*yarn',r'freq_scale_train\s*=\s*0\.03125',
                                r'n_ctx_orig_yarn\s*=\s*8192',r'block_size=7',r'n_max=7']:
                    if not re.search(pattern,text): raise ValueError('The loaded draft settings differ: '+pattern)
            yield url,log_path,command
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: child.kill(); child.wait()


def run_task(cfg, item, task, log_path, no_draft=False):
    records=[]
    started=time.monotonic()
    def call(messages,tools,phase):
        count=evaluation.count_tokens(cfg,messages,tools)
        if count+2048+512 > 65536: raise ValueError('The request exceeds the context reserve.')
        offset=log_path.stat().st_size
        result=evaluation.stream(cfg,messages,tools)
        result['phase']=phase
        records.append(result) # Preserve evidence before each validation.
        with log_path.open('rb') as stream:
            stream.seek(offset)
            text=stream.read().decode('utf-8',errors='replace')
        emitted=[int(x) for x in re.findall(r'process_toke:.*next token:\s*(\d+)',text)]
        result['output_token_ids_from_log']=emitted or None
        result['output_token_ids_verified']=bool(emitted) and len(emitted)==result['usage'].get('completion_tokens')
        if emitted and not result['output_token_ids_verified']:
            raise ValueError('The emitted-token log count differs from reported completion tokens.')
        caps=[int(x) for x in re.findall(r'max possible draft: (\d+)',text)]
        k=cfg['request_options']['speculative.n_max']
        result['observed_draft_caps']=sorted(set(caps))
        if not no_draft and (not caps or max(caps)!=k):
            raise ValueError('The runtime log does not confirm the requested draft cap.')
        if abs(result['usage'].get('prompt_tokens',-100000)-count)>64:
            raise ValueError('The measured prompt count differs from the fixture.')
        if phase=='initial' and result['server_timings'].get('cache_n')!=0:
            raise ValueError('The cold initial request reused prompt state.')
        if k and not all(key in result['server_timings'] for key in ['draft_n','draft_n_accepted']):
            raise ValueError('The response omitted draft timing counters.')
        return result
    try:
        first=call(item['messages'],item['tools'],'initial')
        if task['id']=='invoice-discount':
            validation=validate_code(first['content']) if first['finish_reason']=='stop' else {'passed':False,'reason':'incomplete'}
        else:
            try:
                tool=tool_result(first)
            except (ValueError,KeyError,TypeError) as exc:
                return {'measurement_valid':True,'success':False,'validation':{'passed':False,'reason':str(exc)},
                        'requests':records,'task_wall_s':time.monotonic()-started}
            assistant={'role':'assistant','content':first['content'],'tool_calls':first['tool_calls']}
            if first['reasoning']: assistant['reasoning_content']=first['reasoning']
            final=call(item['messages']+[assistant,tool],TOOLS,'appended_tool_result')
            validation={'passed':evaluation.answer_correct(final,703),'evaluated':True}
        return {'measurement_valid':True,'success':validation.get('passed') is True,'validation':validation,
                'requests':records,'task_wall_s':time.monotonic()-started}
    except Exception as exc:
        return {'measurement_valid':False,'success':False,'error':str(exc),'requests':records,
                'task_wall_s':time.monotonic()-started}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contexts',type=int,nargs='+',default=[3072])
    p.add_argument('--ks',type=int,nargs='+',default=[4,7])
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--port',type=int,default=18125)
    p.add_argument('--proposal',type=Path,required=True,help='Freeze this candidate proposal before any holdout output.')
    p.add_argument('--no-draft-control',action='store_true')
    p.add_argument('--reasoning-budget',type=int,choices=[512],default=None)
    p.add_argument('--run',action='store_true')
    p.add_argument('--exclusive-grant')
    args=p.parse_args()
    if not set(args.contexts)<={3072,8192,32768} or not set(args.ks)<={0,2,4,7} or not 1<=args.repeats<=3:
        p.error('Use bounded contexts, legal K values, and one through three repeats.')
    if len(set(args.contexts))!=len(args.contexts) or len(set(args.ks))!=len(args.ks):
        p.error('Duplicate contexts or K values are not permitted.')
    validator_version=getattr(code_validator,'VALIDATOR_VERSION','invoice-restricted-v1')
    validator_sha256=hashlib.sha256(Path(code_validator.__file__).read_bytes()).hexdigest()
    profile='practical_reasoning512_output2048' if args.reasoning_budget==512 else 'default_reasoning_output2048'
    manifest=json.loads((HERE/'holdout-manifest.json').read_text())
    expected=json.loads((HERE/'fixture-hashes.json').read_text())['holdout']
    if digest(manifest)!=expected:
        raise ValueError('The frozen holdout manifest hash changed.')
    proposal=json.loads(args.proposal.read_text())
    decision=proposal.get('decision',{})
    ranking=decision.get('rankings',{}).get('speculation',{}).get('trial_order')
    if decision.get('mode')!='experiment' or decision.get('auto_promoted') is not False or len(ranking or [])!=4 or set(ranking or [])!={'k0','k2','k4','k7'}:
        raise ValueError('Supply a bounded experiment ranking for all four candidates.')
    ranked_ks=[int(k[1:]) for k in ranking if int(k[1:]) in args.ks]
    selected_k=int(ranking[0][1:])
    schedule=[{'context':size,'repeat':repeat,'task':task['id'],'k':k} for size in args.contexts for repeat in range(args.repeats)
              for task in manifest['tasks'] for k in (ranked_ks if repeat%2==0 else list(reversed(ranked_ks)))]
    if not args.run:
        controls=len(args.contexts)*args.repeats*len(manifest['tasks']) if args.no_draft_control else 0
        print(json.dumps({'schedule':schedule,'candidate_tasks':len(schedule),'no_draft_tasks':controls,'tasks':len(schedule)+controls,'server_max_k':7,'output_limit':2048,
                          'proposal_sha256':digest(proposal),'profile':profile,'reasoning_budget_tokens':args.reasoning_budget,'ranked_candidate_k':selected_k,'GPU_started':False},indent=2));return
    if not args.exclusive_grant: raise ValueError('The root must grant an exclusive GPU slot.')
    args.output.mkdir(parents=True,exist_ok=False)
    # Freeze the proposal and task definitions before any model output exists.
    (args.output/'frozen-proposal.json').write_text(json.dumps(proposal,indent=2))
    (args.output/'frozen-holdout.json').write_text(json.dumps(manifest,indent=2))
    corpus=(backend.ROOT/'Documents/spark-orchestration/drafter-paired-full/corpus.txt').read_text()
    (args.output/'provenance.json').write_text(json.dumps({'grant':args.exclusive_grant,'schedule':schedule,
        'holdout_sha256':digest(manifest),'proposal_sha256':digest(proposal),'corpus_sha256':digest(corpus),
        'output_limit':2048,'validator_version':validator_version,'validator_sha256':validator_sha256,'profile':profile,'reasoning_budget_tokens':args.reasoning_budget,'model_provenance':json.loads((backend.ROOT/'Documents/spark-orchestration/backend-provenance.json').read_text()),
        'draft_provenance':json.loads((backend.ARTIFACTS/'bonsai2-v2-Q4_K_M-yarn32.manifest.json').read_text())},indent=2))
    def interrupted(signum,frame): raise KeyboardInterrupt('The owned experiment was interrupted.')
    signal.signal(signal.SIGTERM,interrupted)
    with evaluation.exclusive(str(backend.ROOT/'Documents/spark-orchestration/gpu-evaluation.lock')):
        fixtures={}
        for no_draft in ([False,True] if args.no_draft_control else [False]):
            with server(args.port,args.output,no_draft) as (url,log_path,command):
                (args.output/('no-draft-command.json' if no_draft else 'draft-command.json')).write_text(json.dumps(command))
                selected=schedule if not no_draft else [dict(row,k=0) for row in schedule if row['k']==ranked_ks[0]]
                for row in selected:
                    task=next(t for t in manifest['tasks'] if t['id']==row['task'])
                    cfg=config(url,row['k'],args.reasoning_budget)
                    key=(row['context'],row['task'])
                    if key not in fixtures:
                        fixtures[key]=fixture(cfg,task,row['context'],corpus)
                        (args.output/f"fixture-{row['context']}-{row['task']}.json").write_text(json.dumps(fixtures[key]))
                    reset=evaluation.command([sys.executable,str(HERE.parent/'agent-eval/llama_hooks.py'),'reset','--url',url],{})
                    if reset.get('cleared') is not True: raise ValueError('The cache reset did not confirm success.')
                    result=run_task(cfg,fixtures[key],task,log_path,no_draft)
                    result.update(row,validator_version=validator_version,validator_sha256=validator_sha256,profile=profile,reasoning_budget_tokens=args.reasoning_budget,ranked_candidate_k=selected_k,proposal_use='Frozen trial order only; no production promotion.',server_mode='no_draft' if no_draft else 'draft_max7',fixture_sha256=fixtures[key]['sha256'])
                    result['model_requests_wall_s']=sum(r['elapsed_s'] for r in result['requests'])
                    result['client_and_validation_wall_s']=result['task_wall_s']-result['model_requests_wall_s']
                    result['output_tokens']=sum(r['usage'].get('completion_tokens',0) for r in result['requests'])
                    result['net_output_tokens_per_wall_s']=result['output_tokens']/result['task_wall_s']
                    with (args.output/'results.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
                    print(json.dumps({k:v for k,v in result.items() if k!='requests'}),flush=True)
                    if not result['measurement_valid']: raise ValueError('The measurement gate failed. Inspect saved evidence before expansion.')


if __name__=='__main__':main()
