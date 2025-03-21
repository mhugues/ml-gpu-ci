# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.    

import os
import numpy as np
import datasets as hf_datasets
from torch.utils.data import IterableDataset
from typing import Dict, Iterable, Union
from transformers import PreTrainedTokenizerBase

class ConcatTokensDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        wrap: bool,
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.should_wrap = wrap

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        mask_buffer = []
        for sample in self.hf_dataset:
            encoded = self.tokenizer(sample['text'],
                                     truncation=True,
                                     padding=False)
            iids = encoded['input_ids']
            mask = encoded['attention_mask']
            buffer = buffer + iids + [self.tokenizer.eos_token_id]
            mask_buffer = mask_buffer + mask + [1]
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                concat_sample_mask = mask_buffer[:self.max_length]
                mask_buffer = mask_buffer[self.max_length:] if self.should_wrap else []
                yield np.array(concat_sample)
