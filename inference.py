import os
import re
import sys
import json
import datetime

import torch
from torch import nn
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from subtask2_args import InferenceArguments, MyArgumentParser
from utils import Seq2SeqDataCollator, Seq2SeqDataset


def submit(args, tokenizer, res, now=True):
    with open('./dialdoc21-sharedtask-phase1/test_subtask2_phase1_ids.json') as f:
        submission = json.load(f)
    for i, sub in enumerate(submission):
        sub["utterance"] = re.sub("<\s>|<s>|<pad>", "", tokenizer.decode(res[i]))
    filename = os.path.join(args.output_dir, args.output_file)
    if now:
        filename += datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(filename + '.json', mode="w", encoding="utf-8") as fp:
        json.dump(submission, fp)


def main():
    parser = MyArgumentParser((InferenceArguments,))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    params = dict(
        pretrained_model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    config = AutoConfig.from_pretrained(**params)
    tokenizer = AutoTokenizer.from_pretrained(**params)
    model = AutoModelForSeq2SeqLM.from_pretrained(config=config, **params)
    
    if args.model_parameters:
        print("====== MODEL PARAMETER LOADING... ======\n"
              f"   {args.model_parameters}")
        model.load_state_dict(torch.load(args.model_parameters))

    max_length = args.test_max_target_length

    # set num_beams for evaluation
    num_beams = args.num_beams if args.num_beams else model.config.num_beams

    test_dataset = Seq2SeqDataset(
        tokenizer=tokenizer,
        type_path='test',
        data_dir=args.data_dir,
        max_target_length=args.test_max_target_length,
        max_source_length=args.max_source_length,
    )

    test_sampler = SequentialSampler(test_dataset)

    data_collator = Seq2SeqDataCollator(tokenizer, args)

    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.per_device_test_batch_size,
        collate_fn=data_collator,
        drop_last=False,
    )

    # prediction_loop
    description = "Prediction"

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)

    print(f"***** Running {description} *****")
    print(f"  Num examples = {num_examples}")
    print(f"  Batch size = {batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    res = []
    for step, inputs in enumerate(test_dataloader):
        # prediction_step, generative based
        has_labels = "labels" in inputs  # False
        # _prepare_inputs
        #  1. device로 보내기
        #  2. memory에 _past 올리기
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        generated_tokens = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            # If PAD token is not defined at least EOS token has to be defined
            padded_tensor = tokenizer.pad_token_id * torch.ones(
                (generated_tokens.shape[0], gen_kwargs["max_length"]),
                dtype=generated_tokens.dtype,
                device=generated_tokens.device,
            )
            padded_tensor[:, :generated_tokens.shape[-1]] = generated_tokens
            generated_tokens = padded_tensor
        loss = None
        labels = None
        res.extend(list(generated_tokens))
    submit(args, tokenizer, res)
    print("Finished!")    
    

if __name__ == '__main__':
    main()
