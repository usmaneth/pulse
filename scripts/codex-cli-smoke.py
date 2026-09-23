#!/usr/bin/env python3
"""Verify the real CLI with a local CPU backend and isolated configuration."""
import argparse,http.server,json,os,pathlib,socket,subprocess,threading,time,urllib.request,re,tempfile
root=pathlib.Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description='Test real Codex routing, resume, and compaction with a CPU mock.')
parser.add_argument('--output',type=pathlib.Path,required=True)
out=parser.parse_args().output
out.mkdir(parents=True,exist_ok=True)
isolated=tempfile.TemporaryDirectory(prefix='bonsai-codex-cli-')
codex_home=pathlib.Path(isolated.name)
requests=[]
def freeport():
 with socket.socket() as s:
  s.bind(('127.0.0.1',0)); return s.getsockname()[1]
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({'model_alias':'cpu-protocol-mock','build_info':'cpu-only','total_slots':1,'default_generation_settings':{'n_ctx':65536}}).encode())
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  requests.append({'path':self.path,'body':body})
  (out/'codex-compaction-cpu-requests.json').write_text(json.dumps(requests,indent=2))
  number=len(requests)
  if number==1:
   delta={'tool_calls':[{'index':0,'id':'probe_call','function':{'name':'exec_command','arguments':json.dumps({'cmd':'printf COMPACTION_TOOL_OK','max_output_tokens':100})}}]}
   finish='tool_calls'
  else:
   delta={'content':'The shell returned COMPACTION_TOOL_OK. The task is complete.'}
   finish='stop'
  events=[{'choices':[{'delta':delta,'finish_reason':None}]},{'choices':[{'delta':{},'finish_reason':finish}]},{'choices':[],'usage':{'prompt_tokens':10000 if number==1 else 100,'completion_tokens':20,'total_tokens':10020 if number==1 else 120}}]
  self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
  for event in events: self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
  self.wfile.write(b'data: [DONE]\n\n')
backend=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=backend.serve_forever,daemon=True).start()
port=freeport()
(codex_home/'config.toml').write_text(f'model = "tripwire"\nmodel_provider = "tripwire"\n[model_providers.tripwire]\nname = "Local tripwire"\nbase_url = "http://127.0.0.1:{backend.server_port}/tripwire/v1"\nwire_api = "responses"\n')
env=dict(os.environ,HOST='127.0.0.1',PORT=str(port),PULSE_BACKEND_URL=f'http://127.0.0.1:{backend.server_port}',PULSE_NATIVE_ENGINE='0',PULSE_CKPT='0')
log=open(out/'codex-compaction-cpu-server.log','w')
adapter=subprocess.Popen(['node','dist/server/index.js'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
try:
 for _ in range(100):
  try:
   urllib.request.urlopen(f'http://127.0.0.1:{port}/ready',timeout=1).close();break
  except Exception:time.sleep(.1)
 taskdir=out/'codex-compaction-cpu-fixture';taskdir.mkdir(exist_ok=True)
 args=[str(root/'scripts/codex-bonsai'),'exec','--skip-git-repo-check','-C',str(taskdir),'-c','model_auto_compact_token_limit=1000','Run printf COMPACTION_TOOL_OK with the shell, then report the result.']
 result=subprocess.run(args,env=dict(os.environ,CODEX_HOME=str(codex_home),BONSAI_CODEX_URL=f'http://127.0.0.1:{port}/v1'),capture_output=True,text=True,timeout=45)
 (out/'codex-compaction-cpu.log').write_text(result.stdout+'\n'+result.stderr)
 print('exit',result.returncode,'backend requests',len(requests))
 assert result.returncode == 0, result.stderr
 assert 'provider: pulse_bonsai' in result.stderr
 session=re.search(r'session id: ([a-z0-9-]+)',result.stderr).group(1)
 resume=subprocess.run([str(root/'scripts/codex-bonsai'),'exec','resume',session,'--skip-git-repo-check','-c','model_auto_compact_token_limit=1000','Report the completed shell result.'],env=dict(os.environ,CODEX_HOME=str(codex_home),BONSAI_CODEX_URL=f'http://127.0.0.1:{port}/v1'),capture_output=True,text=True,timeout=45)
 (out/'codex-resume-cpu.log').write_text(resume.stdout+'\n'+resume.stderr)
 print('resume exit',resume.returncode,'backend requests',len(requests))
 assert resume.returncode == 0, resume.stderr
 assert 'provider: pulse_bonsai' in resume.stderr
 for override in ['model_provider="all-models"','"model_providers"."pulse_bonsai".base_url="http://127.0.0.1:8790/v1"','model="kimi-k3"']:
  blocked=subprocess.run([str(root/'scripts/codex-bonsai'),'exec','--config',override,'hello'],env=dict(os.environ,CODEX_HOME=str(codex_home),BONSAI_CODEX_URL=f'http://127.0.0.1:{port}/v1'),capture_output=True,text=True,timeout=5)
  assert blocked.returncode != 0 and 'cannot override' in blocked.stderr, blocked.stderr
 print('routing override rejection passed')
 assert all(request['path']=='/v1/chat/completions' for request in requests), 'The CLI used the tripwire provider'
 assert any('CONTEXT CHECKPOINT COMPACTION' in str(request['body']) for request in requests), 'No compaction request observed'
 for index,request in enumerate(requests):
  messages=request['body'].get('messages',[])
  print(index+1,request['path'],'messages',len(messages),'last role',messages[-1].get('role'),'last text',str(messages[-1].get('content'))[:220])
finally:
 adapter.terminate();adapter.wait(timeout=10);backend.shutdown();log.close();isolated.cleanup()
