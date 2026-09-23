"""Test complete native state snapshot validation with small synthetic files."""
import contextlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import compare_native_dumps


class NativeDumpTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.roots = [Path(temporary.name)/name for name in ('baseline', 'candidate')]
        names = {'logits.f32'} | {f'layer-{i}.f32' for i in range(64)}
        for i in range(64):
            names |= ({f'key-{i}.f32', f'value-{i}.f32'} if i % 4 == 3 else
                      {f'state-{i}.f32', f'conv-{i}.f32'})
        for root in self.roots:
            step = root/'token-0'
            step.mkdir(parents=True)
            (root/'inputs.txt').write_text('100\n')
            (step/'token.txt').write_text('100\n')
            for name in names:
                (step/name).write_bytes(struct.pack('<3f', 1, 2, 3))

    def compare(self, *options):
        with patch.object(sys, 'argv', ['compare', *map(str, self.roots), *options]), contextlib.redirect_stdout(io.StringIO()):
            return compare_native_dumps.main()

    def test_equal_complete_state(self):
        self.assertEqual(self.compare(), 0)

    def test_changed_recurrent_state(self):
        (self.roots[1]/'token-0'/'state-0.f32').write_bytes(struct.pack('<3f', 1, 2, 4))
        self.assertEqual(self.compare(), 1)

    def test_missing_convolution_in_both_modes(self):
        for root in self.roots:
            (root/'token-0'/'conv-0.f32').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.compare()

    def test_invalid_token_record(self):
        (self.roots[1]/'token-0'/'token.txt').write_text('101\n')
        with self.assertRaisesRegex(ValueError, 'Token record'):
            self.compare()

    def test_explicit_older_baseline_subset(self):
        (self.roots[0]/'token-0'/'conv-0.f32').unlink()
        self.assertEqual(self.compare('--baseline-subset'), 0)


if __name__ == '__main__':
    unittest.main()
