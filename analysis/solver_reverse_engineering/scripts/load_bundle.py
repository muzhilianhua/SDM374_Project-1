"""Local inspection loader for the unmodified bundled Python modules."""
from pathlib import Path
import importlib.abc
import importlib.util
import importlib.machinery
import json
import marshal
import os
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'bundle'
SOURCE = ROOT.parent / 'development-solver-20261112' / 'solver.exe'
_dll_handles = []

def install():
    sys.path.insert(0, str(BUNDLE))
    if hasattr(os, 'add_dll_directory'):
        for folder in (BUNDLE/'torch/lib', BUNDLE/'torch'):
            _dll_handles.append(os.add_dll_directory(str(folder)))
    raw = SOURCE.read_bytes()
    cookie = raw.rfind(b'MEI\x0c\x0b\x0a\x0b\x0e')
    _, length, offset, size, pyver, _ = struct.unpack('!8sIIII64s', raw[cookie:cookie+88])
    assert pyver == 311
    start = cookie + 88 - length
    entries = json.loads((ROOT/'manifest.json').read_text())['entries']
    p, c, *_ = entries['PYZ.pyz']
    pyz = raw[start+p:start+p+c]
    toc = dict(marshal.loads(pyz[struct.unpack('!I', pyz[8:12])[0]:]))
    class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in toc:
                return None
            normal = importlib.machinery.PathFinder.find_spec(fullname, path)
            if normal is not None and normal.loader is not None and normal.loader.__class__.__name__ != 'NamespaceLoader':
                return None
            return importlib.util.spec_from_loader(fullname, self, is_package=bool(toc[fullname][0]))
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            pkg, p, c = toc[module.__name__]
            code = marshal.loads(zlib.decompress(pyz[p:p+c]))
            module.__file__ = str(BUNDLE.joinpath(*module.__name__.split('.')) / '__init__.pyc') if pkg else str(BUNDLE.joinpath(*module.__name__.split('.')).with_suffix('.pyc'))
            if pkg:
                module.__path__ = [str(Path(module.__file__).parent)]
            exec(code, module.__dict__)
    sys.meta_path.insert(0, Finder())
    import certifi
    certifi.where = lambda: str(BUNDLE / 'certifi' / 'cacert.pem')
    return raw, start, entries, pyz, toc

if __name__ == '__main__':
    install()
    import certifi
    print('certifi', certifi, getattr(certifi, '__file__', None), hasattr(certifi, 'where'))
    import citylearn.citylearn
    print('CityLearn import succeeded')
