import os


def traverse_folder(folder_path, is_recursive, filter_func):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path {folder_path} does not exist.")
    if is_recursive:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not filter_func:
                    yield os.path.join(root, file)
                else:
                    if filter_func(file):
                        yield os.path.join(root, file)
    else:
        for file in os.listdir(folder_path):
            if not filter_func:
                yield os.path.join(folder_path, file)
            else:
                if filter_func(file):
                    yield os.path.join(folder_path, file)

def traverse_with_mirror_folder(read_folder_path: str, mirror_folder_path: str, is_recursive, filter_func):
    for filepath in traverse_folder(read_folder_path, is_recursive, filter_func):
        rel_path = os.path.relpath(filepath, read_folder_path)
        mirror_path = os.path.join(mirror_folder_path, rel_path)
        yield filepath, mirror_path

def traverse_and_process_with_mirror_folder(read_folder_path, mirror_folder_path, is_recursive, filter_func, process_func):
    for filepath, mirror_path in traverse_with_mirror_folder(read_folder_path, mirror_folder_path, is_recursive, filter_func):
        if not os.path.exists(os.path.dirname(mirror_path)):
            os.makedirs(os.path.dirname(mirror_path), exist_ok=True)
        process_func(filepath, mirror_path)


class FilenameFilterFunctions:

    @staticmethod
    def suffix_filter(suffix):
        return lambda x: x.endswith(suffix)

    @staticmethod
    def prefix_filter(prefix):
        return lambda x: x.startswith(prefix)

    @staticmethod
    def contains_filter(substring):
        return lambda x: substring in x

    @staticmethod
    def regex_filter(regex):
        import re
        return lambda x: re.match(regex, x)

