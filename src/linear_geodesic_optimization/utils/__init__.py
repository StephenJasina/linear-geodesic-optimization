import sys
from .notebook_import import NotebookFinder

sys.meta_path.append(NotebookFinder())
