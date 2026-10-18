"""Remove machine-specific owner headers from a built source distribution."""
import argparse
import copy
import os
from pathlib import Path
import tarfile
import tempfile


def normalize(path):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tar.gz')
    os.close(descriptor)
    try:
        with tarfile.open(path, 'r:gz') as source, tarfile.open(temporary, 'w:gz', format=tarfile.PAX_FORMAT) as target:
            for original in source.getmembers():
                if original.name.startswith('/') or '..' in Path(original.name).parts:
                    raise ValueError('Unsafe archive path')
                if not (original.isfile() or original.isdir()):
                    raise ValueError('Only regular files and directories are supported')
                member = copy.copy(original)
                member.uid = member.gid = 0
                member.uname = member.gname = ''
                member.pax_headers = {k: v for k, v in original.pax_headers.items() if k not in {'uid', 'gid', 'uname', 'gname', 'atime', 'ctime'}}
                target.addfile(member, source.extractfile(original) if original.isfile() else None)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    normalize(parser.parse_args().archive)
