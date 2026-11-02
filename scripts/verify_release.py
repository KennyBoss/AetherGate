"""Verify frozen inputs and the shipped runtime without training neural weights."""
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_assets():
    assets = json.loads((ROOT / 'release/frozen-assets.json').read_text())
    for name, expected in assets.items():
        path = ROOT / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError('Frozen release input changed: ' + name)
    return assets


def main():
    assets = verify_assets()
    attempts = []
    forbidden_root = '/Users/makbuk/000cor/'
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if path.startswith(forbidden_root):
                raise RuntimeError('Release tried to access the original working tree: ' + path)
        if event == 'subprocess.Popen':
            command = args[1]
            if any(Path(str(v)).name in ('train_ssm_text.py', 'run_stream_memory.py') for v in command):
                raise RuntimeError('Training subprocess is forbidden in release CI')
    sys.addaudithook(audit)
    names = ['test_semantic_memory', 'test_durable_memory'] + sorted(p.stem for p in ROOT.glob('test_codepy*.py'))
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    import train_ssm_text as core
    originals = [getattr(core, name) for name in ('adam_update', 'train_step')]
    def forbidden(*args, **kwargs):
        attempts.append('optimizer update')
        raise RuntimeError('Neural weight training is forbidden in release CI')
    # Legacy interpreter dependencies import these functions by value.
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for name, value in list(vars(module).items()):
            if any(value is original for original in originals):
                setattr(module, name, forbidden)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    verify_assets()
    imported = sorted({str(Path(m.__file__).resolve().relative_to(ROOT))
                       for m in list(sys.modules.values())
                       if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(ROOT)
                       and str(m.__file__).endswith('.py')})
    output = ROOT / '.ci-output'
    output.mkdir(exist_ok=True)
    summary = {'successful': result.wasSuccessful() and not attempts,
               'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'neural_training_attempts': attempts,
               'frozen_inputs_verified': len(assets), 'original_worktree_access_forbidden': True,
               'imported_sources': imported}
    (output / 'release-verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    if not summary['successful']:
        raise SystemExit(1)
    print('RELEASE OK — frozen inputs unchanged; no neural training', flush=True)


if __name__ == '__main__':
    main()
