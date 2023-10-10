# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import os
import hashlib
import time
from . import raw_datasets_sft


def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    if "local/jsonfile" in dataset_name:
        chat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir,
                         os.path.pardir, os.path.pardir))
        if not (os.path.isfile(chat_path + '/data/train.json')
                and os.path.isfile(chat_path + '/data/eval.json')):
            raise RuntimeError(
                f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
            )
        return raw_datasets_sft.LocalJsonFileDataset(output_path, seed, local_rank,
                                                 dataset_name, chat_path)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )

def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset

    def __len__(self):
        length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if "labels" in self.prompt_dataset[idx]:
            return  {
                "input_ids": self.prompt_dataset[idx]["input_ids"],
                "labels": self.prompt_dataset[idx]["labels"]
            }
        else:
            return {
                "input_ids": self.prompt_dataset[idx]["input_ids"],
                "labels": self.prompt_dataset[idx]["input_ids"]
            }


def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chat_history = raw_dataset.get_chat_history(tmp_data)  # the accept response
            if chat_history is not None:
                # modify THIS
                sample = {
                    "input_ids": [],
                    "labels": [],
                }
                for sentence in chat_history:
                    # add role before content
                    sentence_content = f"{sentence['role']}: "+ sentence['content']
                    # sentence_content = sentence['content']

                    # add eos token
                    tokenizer.add_bos_token = True
                    tokenizer.add_eos_token = True

                    sentence_token = tokenizer.encode(sentence_content)
                    if (len(sample["input_ids"]) + len(sentence_token)) > max_seq_len:
                        break
                    sample['input_ids'] += sentence_token
                    if sentence['loss_mask']:
                        sample['labels'] += [-100] * len(sentence_token)
                    else:
                        sample['labels'] += sentence_token
                # padding
                if len(sample['input_ids']) < max_seq_len:
                    sample['input_ids'] += [0] * (max_seq_len - len(sample['input_ids']))
                    sample['labels'] += [-100] * (max_seq_len - len(sample['labels']))
                prompt_dataset.append(sample)
    else:
        raise NotImplementedError
    return PromptDataset(prompt_dataset)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()

    # use all data instead of split for 3-stage
    train_index = get_shuffle_idx(seed, len(train_dataset)).tolist()
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_shuffle_idx(seed, len(eval_dataset)).tolist()
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)

    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          reload=False):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    data_path = data_path[0]
    fname = "_".join(data_path.split("/"))
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        train_dataset, eval_dataset = create_dataset(
            local_rank, data_path, data_split, output_path, train_phase,
            seed, tokenizer, end_of_conversation_token, max_seq_len)

        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    time.sleep(5)
    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)

