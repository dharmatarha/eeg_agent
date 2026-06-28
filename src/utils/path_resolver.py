import os

def resolve_host_path(path: str) -> str:
    """
    If path starts with /mnt/data/, map it to the corresponding host-side path
    using EEG_DATA_DIR or the default project 'data' directory.
    """
    if not path:
        return path
        
    if path.startswith("/mnt/data/"):
        data_dir = os.environ.get("EEG_DATA_DIR")
        if data_dir:
            data_dir = os.path.abspath(data_dir)
        else:
            # Since this file is in src/utils, project root is 2 levels up
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            data_dir = os.path.join(project_root, "data")
        
        rel_part = path[len("/mnt/data/"):]
        return os.path.abspath(os.path.join(data_dir, rel_part))
        
    elif path == "/mnt/data":
        data_dir = os.environ.get("EEG_DATA_DIR")
        if data_dir:
            return os.path.abspath(data_dir)
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            return os.path.join(project_root, "data")
            
    return path
