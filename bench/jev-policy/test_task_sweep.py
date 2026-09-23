import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import task_sweep


class TaskSweepTest(unittest.TestCase):
    def test_tool_execution_validates_actual_arguments(self):
        first={'finish_reason':'tool_calls','tool_calls':[{'id':'c','function':{'name':'multiply','arguments':'{"a":37,"b":19}'}}]}
        self.assertEqual(task_sweep.tool_result(first)['content'],'{"result": 703}')
        first['tool_calls'][0]['function']['arguments']='{"a":37,"b":18}'
        with self.assertRaises(ValueError):task_sweep.tool_result(first)

    def test_incomplete_code_does_not_count_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'server.log';log.write_text('')
            def stream(*args):
                with log.open('a') as out:out.write('max possible draft: 4\n')
                return {'finish_reason':'length','content':'def invoice_total(rows):','usage':{'prompt_tokens':3072},
                        'server_timings':{'cache_n':0,'draft_n':10,'draft_n_accepted':5},'reasoning':'','tool_calls':[]}
            with patch.object(task_sweep.evaluation,'stream',side_effect=stream),patch.object(task_sweep.evaluation,'count_tokens',return_value=3072):
                result=task_sweep.run_task(task_sweep.config('http://unused',4),{'messages':[],'tools':[]},{'id':'invoice-discount'},log)
            self.assertTrue(result['measurement_valid'])
            self.assertFalse(result['success'])
            self.assertEqual(result['validation']['reason'],'incomplete')

    def test_wrong_tool_is_quality_failure_with_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'server.log';log.write_text('')
            def stream(*args):
                with log.open('a') as out:out.write('max possible draft: 7\n')
                return {'finish_reason':'stop','content':'703','usage':{'prompt_tokens':3072},
                        'server_timings':{'cache_n':0,'draft_n':10,'draft_n_accepted':5},'reasoning':'','tool_calls':[]}
            with patch.object(task_sweep.evaluation,'stream',side_effect=stream),patch.object(task_sweep.evaluation,'count_tokens',return_value=3072):
                result=task_sweep.run_task(task_sweep.config('http://unused',7),{'messages':[],'tools':[]},{'id':'tool-multiply'},log)
            self.assertTrue(result['measurement_valid'])
            self.assertFalse(result['success'])
            self.assertEqual(len(result['requests']),1)



class BudgetProfileTest(unittest.TestCase):
    def test_reasoning_budget_is_explicit_and_separate(self):
        base=task_sweep.config('http://unused',4)
        practical=task_sweep.config('http://unused',4,512)
        self.assertNotIn('reasoning_budget_tokens',base['request_options'])
        self.assertEqual(practical['request_options']['reasoning_budget_tokens'],512)
        self.assertEqual(practical['max_tokens'],2048)
        self.assertEqual(practical['request_options']['speculative.n_max'],4)

if __name__=='__main__':unittest.main()
