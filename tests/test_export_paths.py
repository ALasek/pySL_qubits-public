import os
import unittest

from src.export.path_utils import get_data_dir


class ExportPathTests(unittest.TestCase):
    def test_data_dir_uses_repo_relative_path(self):
        data_dir = get_data_dir("subdir")
        self.assertTrue(data_dir.endswith(os.path.join("data", "subdir")))


if __name__ == "__main__":
    unittest.main()
