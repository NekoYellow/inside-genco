"""Checks for data-boundary and integrity failures that would invalidate evaluation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from baseline import load_splits
from download_assets import verify


class AssetTests(unittest.TestCase):
    def test_overlapping_splits_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'splits.json'
            path.write_text(json.dumps(dict(train=[0, 1], val=[2], test=[1, 3])))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_splits(path)

    def test_same_size_corruption_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'asset'
            path.write_bytes(b'good')
            entry = dict(size=4, sha256=hashlib.sha256(b'good').hexdigest())
            self.assertTrue(verify(path, entry))
            path.write_bytes(b'evil')
            self.assertFalse(verify(path, entry))

    def test_git_blob_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config'
            path.write_bytes(b'abc')
            entry = dict(size=3, sha256=None, git_blob=hashlib.sha1(b'blob 3\0abc').hexdigest())
            self.assertTrue(verify(path, entry))
            path.write_bytes(b'xyz')
            self.assertFalse(verify(path, entry))


if __name__ == '__main__':
    unittest.main()
