"""Warning-free CLI entry: python -m telecom_aut.testing <tasks-dir> [...]"""

import sys

from .runner import main

sys.exit(main())
