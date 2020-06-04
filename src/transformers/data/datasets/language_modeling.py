import logging
import os
import pickle
import time
import glob

import tensorflow as tf
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


_PROVIDED_FEATURES = {
    'set_id': tf.io.FixedLenFeature([], tf.int64),
    'index': tf.io.FixedLenFeature([], tf.int64),
    'visit_count_1y': tf.io.FixedLenFeature([], tf.int64),
    'studier_count': tf.io.FixedLenFeature([], tf.int64),
    'content': tf.io.FixedLenFeature([], tf.string),
    'def_lang': tf.io.FixedLenFeature([], tf.string),
    'term_lang': tf.io.FixedLenFeature([], tf.string),
}


class MyDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 batch_size: int,
                 prefetch_size: int,
                 cache_path: str = '/data/cache/roberta_lm_cache',
                 pre_map_cache_path: str = '/data/cache/roberta_lm_cache_premap',
                 debug: bool = False):
        logger.info("Creating dataset from %s", file_path)

        tf.config.set_visible_devices([], 'GPU')
        dataset = MyDataset._my_input_fn(filename_or_glob=file_path,
                                         features=_PROVIDED_FEATURES,
                                         cache_path=cache_path,
                                         pre_map_cache_path=pre_map_cache_path,
                                         batch_size=batch_size,
                                         debug=debug,
                                         prefetch_size=prefetch_size,
                                         )
        self.dataset = dataset.repeat(-1)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.current_skip = 0

    def __len__(self):
        return 187_000_000

    def __getitem__(self, i) -> torch.tensor:
        try:
            self.dataset = self.dataset.skip(max(0, i - self.current_skip))
            self.current_skip = i
            it = iter(self.dataset)
            lines = next(it)
            lines = [line.decode('utf-8') for line in lines.numpy()]
            batch_encoding = self.tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=self.block_size)
            batches = batch_encoding["input_ids"]
        except Exception as e:
            logger.warning('Exception with {}'.format(e))
        return torch.tensor(batches[0], dtype=torch.long)

    @staticmethod
    def _my_parse(serialized_example, features):
        example = tf.io.parse_single_example(serialized_example, features)
        return example['content']

    @staticmethod
    def _my_input_fn(filename_or_glob,
                     features,
                     cache_path='/data/cache/tf_dataset_cache',
                     pre_map_cache_path='/data/cache/tf_dataset_pre_map_cache',
                     delete_cache=False,
                     batch_size=32,
                     prefetch_size=32,
                     debug=False):
        if delete_cache:
            file_list = glob.glob(cache_path + '*')
            for file_path in file_list:
                os.remove(file_path)

        dataset = tf.data.Dataset.list_files(filename_or_glob, shuffle=False)

        def _mapper(x):
            return MyDataset._my_parse(x, features)

        if debug:
            dataset = dataset.interleave(tf.data.TFRecordDataset) \
                .map(_mapper)
        else:
            dataset = dataset.interleave(tf.data.TFRecordDataset,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                         deterministic=True) \
                .map(_mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .prefetch(buffer_size=prefetch_size)
        return dataset
