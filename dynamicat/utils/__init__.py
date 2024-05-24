from dynamicat.utils.cpu_parallel import mproc_map, threads_pool_map
from dynamicat.utils.traverse import traverse_folder, FilenameFilterFunctions


def traverse_files_with_suffix(folder_path, suffix, is_recursive=True):
    return traverse_folder(folder_path, is_recursive, FilenameFilterFunctions.suffix_filter(suffix))

