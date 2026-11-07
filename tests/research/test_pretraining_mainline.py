"""Research launch safety: no accidental retraining or output overwrite."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('pretraining_mainline', ROOT / 'scripts/research/run_pretraining_mainline.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MainlineLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/research/pretraining_mainline.json').read_text(encoding='utf-8'))
        cls.test_root = ROOT / 'outputs/experiments'

    def test_relative_paths_use_repository_root(self):
        self.assertEqual(MODULE.resolve_input(ROOT, self.config['data']), ROOT / self.config['data'])
        with self.assertRaises(ValueError):
            MODULE.resolve_input(ROOT, '../another-repository')

    def test_refuses_existing_outputs(self):
        with self.assertRaises(FileExistsError):
            MODULE.training_commands('fivefold', ROOT, self.config, self.config['fivefold_results'], 'cpu')

    def test_refuses_outputs_in_data_or_outside_repository(self):
        for path in ['data/processed/should-not-write', '../should-not-write']:
            with self.assertRaises(ValueError):
                MODULE.training_commands('fivefold', ROOT, self.config, path, 'cpu')

    def test_training_commands_are_portable_and_do_not_create_output(self):
        with tempfile.TemporaryDirectory(dir=self.test_root) as temporary:
            output = Path(temporary) / 'not-created'
            for action in ['fivefold', 'sixfold', 'multiseed']:
                commands = MODULE.training_commands(action, ROOT, self.config, str(output), 'cpu')
                self.assertFalse(output.exists())
                self.assertTrue(commands)
                for command in commands:
                    self.assertTrue(Path(command[1]).is_relative_to(ROOT / 'scripts/research'))
                    self.assertNotIn('worktrees', ' '.join(command))
                    self.assertNotIn('/home/', ' '.join(command))
                    self.assertEqual(command[command.index('--device')+1], 'cpu')


if __name__ == '__main__':
    unittest.main()
