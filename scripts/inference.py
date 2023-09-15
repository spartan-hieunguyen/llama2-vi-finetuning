import os
from functools import partial

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    pipeline
)
from peft import PeftModel

from argument.argument_class import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from dataset import Prompter, generate_prompt


def main():
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = getattr(torch, training_args.compute_dtype)
    device_map = "cpu" if training_args.no_cuda else "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // world_size


    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=compute_dtype,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    prompter = Prompter(data_args.prompt_template_name,
                        template_json_path=data_args.prompt_path,
                        is_chat_model=training_args.is_chat_model)
    formatting_prompts_func = partial(generate_prompt, prompter=prompter)

    model = PeftModel.from_pretrained(model, training_args.output_dir)

    generation_config = GenerationConfig(
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
        num_beams=3,
        use_cache=True,
        temperature=0.1,
        top_p=0.75,
        repetition_penalty=1.1
    )
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config)
    while True:
        try:
            print("instruction: ")
            instruction = input()
            print("input: ")
            input = input()
            data = {"instruction": instruction, "input": input, "output": None}
            prompt = formatting_prompts_func(data)["text"]
            print("-"*50)
            result = pipe(prompt["text"])
            print(result[0]['generated_text'])
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
