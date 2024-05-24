from dynamicat.utils.traverse import traverse_with_mirror_folder, FilenameFilterFunctions, traverse_folder, \
    traverse_and_process_with_mirror_folder
import os

if __name__ == '__main__':
    test_folder_base = os.path.dirname(os.path.dirname(__file__))
    print(test_folder_base)

    print(list(traverse_with_mirror_folder(
        os.path.join(test_folder_base, "test_traverse_folder"),
        os.path.join(test_folder_base, "test_traverse_folder_mirror"),
        True,
        FilenameFilterFunctions.contains_filter("a")
    )))
    print(list(traverse_folder(
        os.path.join(test_folder_base, "test_traverse_folder"),
        True,
        None
    )))

    print(list(traverse_folder(
        os.path.join(test_folder_base, "test_traverse_folder_mirror"),
        True,
        None
    )))

    def process_func(filepath, mirror_path):
        with open(filepath, "r") as f:
            content = f.read()
        with open(mirror_path, "w") as f:
            f.write(content)

    traverse_and_process_with_mirror_folder(
        os.path.join(test_folder_base, "test_traverse_folder"),
        os.path.join(test_folder_base, "test_traverse_folder_mirror"),
        True,
        FilenameFilterFunctions.contains_filter("a"),
        process_func
    )
