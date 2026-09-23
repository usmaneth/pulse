#!/usr/bin/env python3
"""Test scoped hook trust and real Codex startup/resume with a CPU mock."""
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import tempfile
import threading
import tomllib
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
MODULE = runpy.run_path(str(ROOT / 'codex-bonsai-continuity.py'))

class ContinuityTest(unittest.TestCase):
    def test_invalid_task_and_hook_overrides_do_not_register(self):
        for task,args,config in [('bad.task',[],[]),('valid',['--worktree'],[]),
                                 ('valid',[],['-c','features."hooks"=false']),
                                 ('valid',['--disable','hooks'],[])]:
            with patch.dict(os.environ,{'BONSAI_CODEX_TASK':task}), patch('subprocess.Popen') as launch:
                with self.assertRaises(ValueError):
                    MODULE['configure']('codex',{},args,config)
                launch.assert_not_called()

    def test_trust_refuses_config_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);other=home/'original';other.write_text('model="keep"\n')
            (home/'config.toml').symlink_to(other)
            with self.assertRaises(OSError):MODULE['register_trust'](home,{'only-ours':'sha256:'+'a'*64})
            self.assertEqual(other.read_text(),'model="keep"\n')

    def test_trust_preserves_config_and_rejects_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory); path=home/'config.toml'
            original=b'# User settings\nmodel_provider="unchanged"\nsandbox_mode="read-only"\n'
            path.write_bytes(original)
            MODULE['register_trust'](home, {'only-ours':'sha256:'+'a'*64})
            once=path.read_bytes();self.assertTrue(once.startswith(original))
            MODULE['register_trust'](home, {'only-ours':'sha256:'+'a'*64})
            self.assertEqual(path.read_bytes(),once)
            with self.assertRaises(ValueError):MODULE['register_trust'](home, {'only-ours':'sha256:'+'b'*64})
            self.assertEqual(path.read_bytes(),once)

    def test_cli_start_and_resume_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory); codex_home=home/'codex';codex_home.mkdir();work=home/'work';work.mkdir();(home/'vault').mkdir()
            config=codex_home/'config.toml'
            original='developer_instructions="KEEP_EXISTING_INSTRUCTIONS"\nsandbox_mode="danger-full-access"\napproval_policy="never"\n'
            spy=home/'hook-spy.py'
            spy.write_text("import json,sys,pathlib\np=pathlib.Path(__file__).with_name('hook-events.jsonl')\ne=json.load(sys.stdin)\nwith p.open('a') as f:f.write(json.dumps({'event':e.get('hook_event_name'),'source':e.get('source')})+'\\n')\nprint('EXISTING_USER_HOOK')\n")
            original+='[[hooks.SessionStart]]\n[[hooks.SessionStart.hooks]]\ntype="command"\ncommand='+json.dumps(shlex.join(['python3',str(spy)]))+'\n' 
            config.write_text(original)
            requests=[]
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_GET(self):
                    self.send_response(200);self.end_headers()
                    self.wfile.write(json.dumps(dict(engine='pulse',backend='http://127.0.0.1:1',context_per_slot=65536,model='mock',backend_build='cpu-test')).encode())
                def do_POST(self):
                    payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(payload)
                    item=dict(id='msg_cpu',type='message',status='completed',role='assistant',content=[dict(type='output_text',text='CPU_OK',annotations=[])])
                    if len(requests)==1:
                        command='printf done >> '+shlex.quote(str(work/'completed-tool.txt'))
                        item=dict(id='fc_cpu',type='function_call',status='completed',call_id='call_cpu',name='exec_command',arguments=json.dumps(dict(cmd=command,max_output_tokens=20)))
                    count=50000 if len(requests)==2 else 100
                    response=dict(id='resp_cpu',object='response',created_at=1,status='completed',model='bonsai-2-27b',output=[item],usage=dict(input_tokens=count,output_tokens=3,total_tokens=count+3))
                    events=[dict(type='response.created',response={**response,'status':'in_progress','output':[]}),dict(type='response.output_item.added',output_index=0,item={**item,'status':'in_progress','content':[]}),dict(type='response.output_text.delta',item_id='msg_cpu',output_index=0,content_index=0,delta='CPU_OK'),dict(type='response.output_item.done',output_index=0,item=item),dict(type='response.completed',response=response)]
                    if len(requests)==1:
                        events=[events[0],dict(type='response.output_item.added',output_index=0,item={**item,'arguments':''}),dict(type='response.function_call_arguments.delta',item_id='fc_cpu',output_index=0,delta=item['arguments']),dict(type='response.output_item.done',output_index=0,item=item),dict(type='response.completed',response=response)]
                    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                    self.wfile.write(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            env=dict(os.environ,HOME=str(home),CODEX_HOME=str(codex_home),BONSAI_CODEX_TASK='cpu-continuity',PULSE_CONTINUITY_JEV='0',BONSAI_CODEX_URL=f'http://127.0.0.1:{server.server_port}/v1')
            with patch.dict(os.environ,env,clear=True):
                existing=MODULE['inspect_hooks']('codex',{'features.hooks':True},work)
                MODULE['register_trust'](codex_home,{h['key']:h['currentHash'] for h in existing if h['source']=='user'})
            original=config.read_text()
            for key in ('BONSAI_CODEX_PROJECT','BONSAI_CODEX_CONTEXT','BONSAI_CODEX_VAULT','CODEX_BIN'):env.pop(key,None)
            try:
                for args in [['exec','--skip-git-repo-check','--json','Reply CPU_OK.'],['exec','resume','--last','--skip-git-repo-check','--json','Reply CPU_OK again.']]:
                    result=subprocess.run([str(ROOT/'codex-bonsai'),*args],env=env,cwd=work,text=True,capture_output=True,timeout=40)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn('CPU_OK',result.stdout)
                self.assertEqual(len(requests),4)
                self.assertEqual((work/'completed-tool.txt').read_text(),'done')
                for request in [requests[0],requests[-1]]:
                    text=json.dumps(request)
                    self.assertIn('KEEP_EXISTING_INSTRUCTIONS',text)
                    self.assertIn('Local task continuity',text)
                    self.assertIn('cpu-continuity',text)
                    self.assertIn('EXISTING_USER_HOOK',text)
                self.assertTrue(config.read_text().startswith(original))
                states=tomllib.loads(config.read_text())['hooks']['state'];self.assertEqual(len(states),4)
                self.assertNotIn('bypass',config.read_text())
                checkpoint=list((home/'.local/state/pulse/continuity').glob('*/cpu-continuity/checkpoint.json'))
                self.assertEqual(len(checkpoint),1)
                snapshot=checkpoint[0].with_name('precompact.json')
                self.assertTrue(snapshot.exists())
                self.assertEqual(json.loads(snapshot.read_text())['revision'],1)
                self.assertIn('Revision: 1',json.dumps(requests[-1]))
                sources=[json.loads(line)['source'] for line in (home/'hook-events.jsonl').read_text().splitlines()]
                self.assertEqual(sources.count('compact'),1)
                self.assertIn('startup',sources)
                self.assertIn('resume',sources)
            finally:
                server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
