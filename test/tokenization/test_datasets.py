from dynamicat.tokenization.filebase_dataset import FileBaseDataset

import os

from test.tokenization.test_dataset_metadata import m3, m5, m6, m7

test_folder_base = os.path.dirname(os.path.dirname(__file__))
print(test_folder_base)
d = FileBaseDataset(
        m5
    )

d2 = FileBaseDataset(
    m6)

d3 = FileBaseDataset(
    m7
)


if __name__ == '__main__':
    for _ in [d, d2, d3]:
        print(_)
        _.load()
        print(len(_))
        # for data in _.iterate():
        #     print(data)
        for _ in _.make_tokenization_configs():
            print(_)
