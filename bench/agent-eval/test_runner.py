import io
import json
import tempfile
import unittest
from unittest.mock import patch
import runner
from code_fixture import validate
from audit_saved import first_diff


class Checks(unittest.TestCase):
    def test_sse_multiline_and_done(self):
        wire = io.BytesIO(b': ping\n\ndata: {"choices":\ndata: []}\n\ndata: [DONE]\n\n')
        self.assertEqual(list(runner.events(wire)), [{"choices": []}, {"_done": True}])

    def test_partial_sse_is_error(self):
        with self.assertRaises(ValueError):
            list(runner.events(io.BytesIO(b'data: {"x": 1}\n')))

    def test_syntax_does_not_pass(self):
        value = {"finish_reason": "tool_calls", "tool_calls": [{"id": "c", "function": {"name": "read_record", "arguments": '{"key":"wrong"}'}}]}
        self.assertFalse(runner.validate_call(value))
        value["tool_calls"][0]["function"]["arguments"] = '{"key":"sku-73"}'
        self.assertTrue(runner.validate_call(value))
        self.assertFalse(runner.answer_correct({"finish_reason": "stop", "tool_calls": [], "content": "{}"}, 2233))
        self.assertFalse(runner.answer_correct({"finish_reason": "length", "tool_calls": [], "content": "2233"}, 2233))

    def test_later_failure_keeps_completed_metrics(self):
        first = {"finish_reason": "tool_calls", "content": "", "usage": {"prompt_tokens": 8192}, "elapsed_s": 1.25, "tool_calls": [{"id": "c", "function": {"name": "read_record", "arguments": '{"key":"sku-73"}'}}]}
        with patch.object(runner, "stream", side_effect=[first, TimeoutError("later timeout")]):
            result = runner.task({}, [], 8192)
        self.assertFalse(result["success"])
        self.assertEqual(result["requests"][0]["elapsed_s"], 1.25)
        self.assertIn("later timeout", result["error"])

    def test_exclusive_lock(self):
        with tempfile.NamedTemporaryFile() as file:
            with runner.exclusive(file.name):
                with self.assertRaises(BlockingIOError):
                    with runner.exclusive(file.name):
                        pass

    def test_expired_lease(self):
        with self.assertRaises(ValueError):
            runner.preflight({}, {"expires_unix": 0})

    def test_code_rejects_syntax_only_and_imports(self):
        self.assertFalse(validate("def invoice_total(rows): return 0")["passed"])
        self.assertFalse(validate("import os")["passed"])

    def test_first_difference(self):
        self.assertIsNone(first_diff([1, 2], [1, 2]))
        self.assertEqual(first_diff([1], [1, 2]), 1)
        self.assertEqual(first_diff([1, 3], [1, 2]), 1)

    def test_tool_output_is_appended(self):
        initial = {"finish_reason": "tool_calls", "content": "", "usage": {"prompt_tokens": 8192}, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "read_record", "arguments": '{"key":"sku-73"}'}}]}
        answer = {"finish_reason": "stop", "content": "2233", "tool_calls": []}
        compact = {"finish_reason": "stop", "content": "2871", "tool_calls": []}
        code = {"finish_reason": "stop", "content": "def invoice_total(rows):\n total = 0\n for row in rows:\n  q = row[\"quantity\"]\n  p = row[\"unit_price_cents\"]\n  d = row.get(\"discount_percent\", 0)\n  if q < 0 or p < 0 or d < 0 or d > 100: raise ValueError()\n  total += q*p*(100-d)//100\n return total"}
        with patch.object(runner, "stream", side_effect=[initial, answer, compact, code]) as mock:
            result = runner.task({}, [{"role": "user", "content": "task"}], 8192)
            self.assertTrue(result["success"])
            appended = mock.call_args_list[1].args[1]
            self.assertEqual(appended[-1]["tool_call_id"], "c")
            self.assertEqual(json.loads(appended[-1]["content"])["unit_price_cents"], 319)


if __name__ == "__main__":
    unittest.main()
