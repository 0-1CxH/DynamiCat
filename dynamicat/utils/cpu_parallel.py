import functools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from loguru import logger
from tqdm import tqdm

def mproc_map(func, items, chunk_size, ordered=True, max_workers=cpu_count()):
    logger.info(f"Using multi processing map ({ordered=}) to {func.__name__} with {len(items)} items({chunk_size=}), parallelism={max_workers}")
    results = []
    with multiprocessing.Pool(processes=max_workers) as pool:
        if ordered:
            converted = pool.imap(func, items, chunksize=chunk_size)
        else:
            converted = pool.imap_unordered(func, items, chunksize=chunk_size)
        with tqdm(total=len(items), unit="record") as progress_bar:
            for idx, current_result in enumerate(converted):
                results.append(current_result)
                progress_bar.update(1)
    return results

def threads_pool_map(func, items, ordered=True, max_workers=cpu_count()):
    logger.info(f"Using threads pool ({ordered=}) to {func.__name__} with {len(items)} items, parallelism={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if ordered:
            futures = {idx: executor.submit(func, item) for idx, item in enumerate(items)}
            with tqdm(total=len(items), unit="record") as progress_bar:
                for idx in range(len(items)):
                    results.append(futures[idx].result())
                    progress_bar.update(1)
        else:
            futures = [executor.submit(func, item) for item in items]
            with tqdm(total=len(items), unit="record") as progress_bar:
                for future in futures:
                    results.append(future.result())
                    progress_bar.update(1)
    return results