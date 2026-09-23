#!/usr/bin/env python3
"""Recheck every saved code output without changing original measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'agent-eval'))
import code_fixture


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('directories',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('The reanalysis destination already exists.')
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'validator_version':code_fixture.VALIDATOR_VERSION,
            'validator_path':code_fixture.__file__,'validator_sha256':sha(Path(code_fixture.__file__)),
            'scope':'CPU reanalysis only. No output regeneration or mutation of original results.','sources':[],'rows':[]}
    for directory in args.directories:
        source=directory/'results.jsonl'
        result['sources'].append({'path':str(source),'sha256':sha(source)})
        for index,line in enumerate(source.read_text().splitlines()):
            row=json.loads(line)
            updated=row['validation']
            if row['task']=='invoice-discount':
                first=row['requests'][0]
                updated=code_fixture.validate(first['content']) if first['finish_reason']=='stop' else {'passed':False,'evaluated':False,'reason':'incomplete'}
            result['rows'].append({'source':str(source),'row_index':index,'task':row['task'],'k':row['k'],
                'profile':row.get('profile','default_reasoning_output2048'),'server_mode':row['server_mode'],
                'original_validation':row['validation'],'updated_validation':updated,
                'code_reanalyzed':row['task']=='invoice-discount','task_wall_s':row['task_wall_s'],
                'updated_success':updated.get('passed') is True})
        if result['sources'][-1]['sha256']!=sha(source):raise ValueError('The original result file changed during reanalysis.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'rows':len(result['rows']),'code_outputs_reanalyzed':sum(r['code_reanalyzed'] for r in result['rows']),
                      'updated_successes':sum(r['updated_success'] for r in result['rows']),
                      'validator_sha256':result['validator_sha256']}))


if __name__=='__main__':main()
