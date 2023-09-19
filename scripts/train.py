import os
from functools import partial

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser
)
from peft import LoraConfig

from dataset import Prompter, generate_prompt
from config.config import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from argument.argument_class import DataTrainingArguments, ModelArguments, CustomTrainingArguments


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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=training_args.use_4bit,
        bnb_4bit_quant_type=training_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=training_args.use_nested_quant,
    )

    if not training_args.no_cuda and compute_dtype == torch.float16 and training_args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        r=training_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[training_args.model_type],
    )

    prompter = Prompter(data_args.prompt_template_name,
                        template_json_path=data_args.prompt_path,
                        is_chat_model=training_args.is_chat_model)
    formatting_prompts_func = partial(generate_prompt, prompter=prompter)

    # Load dataset
    if data_args.train_data_file:
        train_dataset = load_dataset("json", data_files=data_args.train_data_file, split="train")
        train_dataset = train_dataset.shuffle().map(formatting_prompts_func)
    if data_args.eval_data_file:
        eval_dataset = load_dataset("json", data_files=data_args.eval_data_file, split="train")
        eval_dataset = eval_dataset.shuffle().map(formatting_prompts_func)

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=training_args.packing,
    )

    # Train model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save trained model
    trainer.model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
