import sys
from pathlib import Path

# Add the project root to sys.path to allow for absolute imports from `lib` and `scripts`.
# The project root is three levels up from this conftest.py file
# (tests/evaluation/conftest.py -> tests/evaluation -> tests -> project_root).
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
