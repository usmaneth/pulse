"""Test checkpoint recovery, conflicts, bounds, and local retrieval."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('local_context', Path(__file__).with_name('local_context.py'))
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'repo'
        self.project.mkdir()
        self.storage = self.root / 'state'
        self.vault = self.root / 'vault'
        (self.vault / 'projects').mkdir(parents=True)
        self.state = {key: [] for key in lc.FIELDS}
        self.state.update(objective='Repair the cedar retry behavior', constraints=['Do not change the public API'])

    def save(self, **kwargs):
        return lc.save(self.project, 'cedar', self.state, state_root=self.storage, **kwargs)

    def test_repeated_updates_resume_and_stale_conflict(self):
        path, first = self.save(initialize=True)
        for revision in range(1, 5):
            self.state['completed'].append(f'Test {revision} passed; evidence tests/retry.py')
            _, value = self.save(expected_revision=revision)
            self.assertEqual(value['revision'], revision + 1)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            self.save(expected_revision=1)
        restored = lc.load(path)
        self.assertEqual(restored['state']['constraints'], self.state['constraints'])
        self.assertEqual(len(restored['state']['completed']), 4)
        self.assertEqual(lc.load(path.parent / 'previous.json')['revision'], 4)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.save(initialize=True)[1], restored)

    def test_invalid_update_preserves_current(self):
        path, _ = self.save(initialize=True)
        old = path.read_bytes()
        self.state['blockers'] = ['x' * 1600]
        with self.assertRaises(ValueError):
            self.save(expected_revision=1)
        self.assertEqual(path.read_bytes(), old)

    def test_vault_body_match_source_lines_and_symlink_exclusion(self):
        note = self.vault / 'projects' / 'design.md'
        note.write_text('# Design\nThe cedar retry must preserve idempotency.\n')
        outside = self.root / 'secret.md'
        outside.write_text('cedar retry secret')
        (self.vault / 'projects' / 'cedar-link.md').symlink_to(outside)
        (self.vault / 'agents').mkdir()
        (self.vault / 'agents' / 'cedar.md').write_text('raw transcript cedar')
        hits = lc.search(self.vault, 'cedar retry')
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['path'], str(note))
        self.assertIn('2: The cedar', hits[0]['excerpt'])
        self.assertEqual(lc.search(self.vault, 'the and for'), [])

    def test_project_isolation_and_traversal(self):
        first, _ = self.save(initialize=True)
        other = self.root / 'other'
        other.mkdir()
        second, _ = lc.save(other, 'cedar', self.state, initialize=True, state_root=self.storage)
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            lc.task_dir(self.project, '../../outside', self.storage)

    def test_context_bound_and_evidence(self):
        self.save(initialize=True)
        text = lc.render(self.project, 'cedar', self.vault, state_root=self.storage)
        self.assertIn('Revision: 1', text)
        self.assertIn('Do not change the public API', text)
        self.assertLessEqual(len(text), lc.MAX_CONTEXT)

    def test_ancestor_symlink_refused_before_write(self):
        self.storage.mkdir()
        outside = self.root / 'outside'
        outside.mkdir()
        digest = lc.hashlib.sha256(str(self.project.resolve()).encode()).hexdigest()[:24]
        (self.storage / digest).symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.save(initialize=True)
        self.assertEqual(list(outside.iterdir()), [])

    def test_all_entries_count_toward_search_budget(self):
        for index in range(20):
            (self.vault / 'projects' / f'dir{index}').mkdir()
        with patch.object(lc.os, 'scandir', wraps=lc.os.scandir) as scanner:
            list(lc.vault_files(self.vault, max_entries=3))
        self.assertLessEqual(scanner.call_count, 4)

    def test_oversized_reads_consume_total_byte_budget(self):
        source = self.root / 'oversized'
        source.write_bytes(b'x' * 65537)
        for index in range(300):
            lc.os.link(source, self.vault / 'projects' / f'note{index}.md')
        with patch.object(lc.os, 'open', wraps=lc.os.open) as opened:
            self.assertEqual(lc.search(self.vault, 'cedar'), [])
        self.assertEqual(opened.call_count, 256)

    def test_jev_cache_skips_duplicate_calls_and_preserves_state(self):
        path, state = self.save(initialize=True)
        response = type('Response', (), dict(returncode=0, stdout=json.dumps(dict(
            advisory_only=True, source='typesafe_native', actions=dict(checkpoint='now')))))()
        with patch.object(lc.subprocess, 'run', return_value=response) as run:
            first = lc.continuity_advice(path.parent, state, 'retry')
            second = lc.continuity_advice(path.parent, state, 'retry')
        self.assertFalse(first['cache_hit'])
        self.assertTrue(second['cache_hit'])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(lc.load(path), state)
        self.assertNotIn('constraints', json.loads(run.call_args.kwargs['input']))


if __name__ == '__main__':
    unittest.main()
