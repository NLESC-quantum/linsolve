"""Version module, kept only for backwards-compatibility."""

import warnings

from . import __version__

version_info = __version__
version = __version__
git_origin = __version__
git_hash = __version__
git_description = __version__
git_branch = __version__

warnings.warn("You should not rely on this module any more. Just use __version__.")


if __name__ == "__main__":
    print(__version__)
