# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset
import os


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class ChatRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The output should be [{"content": "xxx", "role": "user/assistant/system", "loss_mask": True}]
    def get_chat_history(self, sample):
        return



class LocalJsonFileDataset(ChatRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train": os.path.join(chat_path, 'train.json'),
                                             "eval": os.path.join(chat_path, 'eval.json'),
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    # The output should be {"chat": [{"content": "xxx", "role": "user/assistant/system", "loss_mask": True}]}
    def get_chat_history(self, sample):
        chat_history = sample.get('chat', [])
        for idx, chat in enumerate(chat_history):
            if "loss_mask" not in chat:
                if chat['role'] in ["system", "user"]:
                    chat['loss_mask'] = True
                else:
                    chat['loss_mask'] = False
                chat_history[idx] = chat
            assert chat.get("role", "") in ['user', 'assistant', 'system'], f"{chat}: role is not supported."
            assert len(chat.get("content", "")) > 0, f"{chat}: content is empty."
        return chat_history
