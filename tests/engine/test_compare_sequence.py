"""Test diagnostic file validation with synthetic tensor fixtures."""
import contextlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import compare_sequence


class SequenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ref, self.native = [Path(self.temp.name)/name for name in ('reference', 'native')]
        values = np.array([1, 2, 3], dtype='<f4').tobytes()
        header = struct.pack('<i4q', 0, 3, 1, 1, 1)
        for root in (self.ref, self.native):
            step = root/'token-0'
            step.mkdir(parents=True)
            (root/'inputs.txt').write_text('100\n')
            (step/'token.txt').write_text('100\n')
            (step/'logits.f32').write_bytes(values)
        for layer in range(64):
            (self.ref/'token-0'/f'l_out-{layer}.bin').write_bytes(header+values)
            (self.native/'token-0'/f'layer-{layer}.f32').write_bytes(values)
            if layer % 4 != 3:
                (self.ref/'token-0'/f'new_state-{layer}.bin').write_bytes(header+values)
                (self.native/'token-0'/f'state-{layer}.f32').write_bytes(values)

    def run_compare(self):
        with patch.object(sys, 'argv', ['compare', str(self.ref), str(self.native)]), contextlib.redirect_stdout(io.StringIO()):
            return compare_sequence.main()

    def test_equal(self):
        self.assertEqual(self.run_compare(), 0)

    def test_stale_steps(self):
        for root in (self.ref, self.native):
            (root/'token-1').mkdir()
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.run_compare()

    def test_input_mismatch(self):
        (self.native/'inputs.txt').write_text('101\n')
        with self.assertRaisesRegex(ValueError, 'manifests differ'):
            self.run_compare()

    def test_legacy_strided_payload(self):
        path = self.ref/'token-0'/'l_out-0.bin'
        path.write_bytes(path.read_bytes()+b'\0'*4)
        with self.assertRaisesRegex(ValueError, 'Noncontiguous legacy'):
            self.run_compare()

    def test_greedy_mismatch_despite_close_logits(self):
        (self.ref/'token-0'/'logits.f32').write_bytes(np.array([1, 2, 2.001], dtype='<f4').tobytes())
        (self.native/'token-0'/'logits.f32').write_bytes(np.array([1, 2.001, 2], dtype='<f4').tobytes())
        self.assertEqual(self.run_compare(), 1)


if __name__ == '__main__':
    unittest.main()
