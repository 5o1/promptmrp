import os
from typing import Set

class FileBoundBlacklist:
    def __init__(self, path):
        """
        Initialize the blacklist and bind it to a file.
        
        :param file_path: str, the path to the file used to persist the blacklist
        """
        self.path = path
        self.blacklist:Set[str] = set()

        # Load existing blacklist from file if it exists
        if os.path.exists(path):
            self._load_from_file()
        else:
            self._init_file()

    def _init_file(self):
        """Create the file if it doesn't exist."""
        with open(self.path, 'w'):
            pass

    def _load_from_file(self):
        """Load blacklist from the file."""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.blacklist.add(line)

    def append(self, item: str):
        """
        Add an item to the blacklist and sync to the file.
        
        :param item: str, the item to add to the blacklist
        """
        item = item.strip()
        if item not in self.blacklist:
            self.blacklist.add(item)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(item + "\n")
    
    def __contains__(self, item: str) -> bool:
        """
        Check if an item is in the blacklist.
        
        :param item: str, the item to check
        :return: bool, True if the item is in the blacklist, False otherwise
        """
        return item in self.blacklist
