import sys
from unittest.mock import MagicMock

# Mock problematic dependencies to allow test collection
sys.modules['langchain_chroma'] = MagicMock()
sys.modules['langchain.storage'] = MagicMock()
sys.modules['langchain_classic.storage'] = MagicMock()
sys.modules['langchain.retrievers'] = MagicMock()
sys.modules['langchain_classic.retrievers'] = MagicMock()


