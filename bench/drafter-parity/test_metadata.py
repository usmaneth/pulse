"""Test the draft metadata conversion with a small CPU fixture."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import gguf
from metadata_variant import convert, file_hash


class MetadataTest(unittest.TestCase):
    def test_payload_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.gguf'
            writer = gguf.GGUFWriter(source, 'dflash')
            writer.add_float32('dflash.rope.freq_base', 10000000)
            writer.add_uint32('dflash.context_length', 262144)
            writer.add_string('general.name', 'fixture')
            writer.add_tensor('test.weight', np.arange(128, dtype=np.float32).reshape(8, 16))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            initial = file_hash(source)
            first = convert(source, root/'first.gguf', gguf)
            second = convert(source, root/'second.gguf', gguf)
            self.assertEqual(first['source_sha256'], initial)
            self.assertEqual(file_hash(source), initial)
            self.assertTrue(first['tensor_payloads_identical'])
            self.assertEqual(first['destination_sha256'], second['destination_sha256'])
            self.assertEqual(first['metadata_changes']['dflash.rope.scaling.attn_factor'], 1)
            with self.assertRaises(ValueError):
                convert(source, root/'first.gguf', gguf)


if __name__ == '__main__':
    unittest.main()
