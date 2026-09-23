#!/usr/bin/env python3
"""Compare measured task outcomes before latency ratios."""
import argparse
import json
from pathlib import Path
import statistics


def transcript(row):
    return [{'content':r['content'],'reasoning':r['reasoning'],'finish_reason':r['finish_reason'],
             'calls':[c['function'] for c in r['tool_calls']]} for r in row['requests']]


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    rows=[json.loads(s) for s in (args.directory/'results.jsonl').read_text().splitlines()]
    result={'scope':'Completed task outcomes. Transcript equality does not establish token-ID equality.',
            'rows':len(rows),'groups':[],'jev_vs_fixed_k4':[]}
    for size in sorted({r['context'] for r in rows}):
        for task in sorted({r['task'] for r in rows}):
            group=[r for r in rows if r['context']==size and r['task']==task]
            for mode,k in sorted({(r['server_mode'],r['k']) for r in group}):
                matched=[r for r in group if r['server_mode']==mode and r['k']==k]
                times=[r['task_wall_s'] for r in matched]
                equality=[]
                for row in matched:
                    baseline=next((b for b in group if b['server_mode']=='draft_max7' and b['k']==0 and b['repeat']==row['repeat']),None)
                    equality.append(transcript(row)==transcript(baseline) if baseline else None)
                result['groups'].append({'context':size,'task':task,'server_mode':mode,'k':k,'trials':len(matched),
                    'successes':sum(r['success'] for r in matched),'valid_measurements':sum(r['measurement_valid'] for r in matched),
                    'wall_median_s':statistics.median(times),'wall_min_s':min(times),'wall_max_s':max(times),
                    'transcript_matches_loaded_draft_k0':equality})
            for repeat in sorted({r['repeat'] for r in group}):
                current=[r for r in group if r['repeat']==repeat and r['server_mode']=='draft_max7']
                fixed=next((r for r in current if r['k']==4),None)
                selected=next((r for r in current if r['k']==r['ranked_candidate_k']),None)
                if fixed and selected:
                    both=bool(fixed['success'] and selected['success'] and fixed['measurement_valid'] and selected['measurement_valid'])
                    result['jev_vs_fixed_k4'].append({'context':size,'task':task,'repeat':repeat,'selected_k':selected['k'],
                        'selected_success':selected['success'],'fixed_success':fixed['success'],
                        'fixed_over_selected_wall_if_both_successful':fixed['task_wall_s']/selected['task_wall_s'] if both else None,
                        'policy_scope':'Frozen candidate ranking counterfactual. No production promotion.'})
    (args.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
