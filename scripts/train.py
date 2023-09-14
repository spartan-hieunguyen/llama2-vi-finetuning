import os
from functools import partial

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments
)
from peft import LoraConfig, set_peft_model_state_dict

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

    if training_args.resume_from_checkpoint:
        checkpoint_name = os.path.join(
            training_args.resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                training_args.resume_from_checkpoint, "adapter_model.bin"
            )
            training_args.resume_from_checkpoint = False

        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        r=training_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # for transformers.Trainer
    # tokenize_fn = partial(tokenize, tokenizer=tokenizer, add_eos_token=True, max_len=512)
    # generate_and_tokenize_prompt = partial(generate_and_tokenize_prompt,
    #                                        tokenize_fn=tokenize_fn,
    #                                        prompter=prompter,
    #                                        train_on_inputs=True,
    #                                        add_eos_token=True)
    # train_dataset.map(generate_and_tokenize_prompt)
    # eval_dataset.map(generate_and_tokenize_prompt)

    prompter = Prompter(data_args.prompt_template_name, template_json_path=data_args.prompt_path)
    formatting_prompts_func = partial(generate_prompt, prompter=prompter)

    # Load dataset
    train_dataset = load_dataset("json", data_files=data_args.train_data_file, split="train")
    eval_dataset = load_dataset("json", data_files=data_args.eval_data_file, split="train")
    train_dataset = train_dataset.shuffle().map(formatting_prompts_func)
    eval_dataset = eval_dataset.shuffle().map(formatting_prompts_func)


    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optim=training_args.optim,
        save_steps=training_args.save_steps,
        logging_steps=training_args.logging_steps,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        max_grad_norm=training_args.max_grad_norm,
        max_steps=training_args.max_steps,
        warmup_ratio=training_args.warmup_ratio,
        group_by_length=training_args.group_by_length,
        lr_scheduler_type=training_args.lr_scheduler_type,
        report_to=training_args.report_to,
        save_total_limit=training_args.save_total_limit
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=training_args.packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
