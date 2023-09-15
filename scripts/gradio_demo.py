import os
from functools import partial

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    StoppingCriteriaList
)
from peft import PeftModel

from argument.argument_class import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from dataset import Prompter, generate_prompt
from demo_ui.demo import demo
from utils.stopping_criteria import StoppingCriteria
from utils.callbacks import Stream, Iteratorize


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=1024,
    stream_output=False,
    generate_prompt_fn=None,
    prompter=None,
    device="auto",
    tokenizer=None,
    model=None,
    stop_criteria=None,
    **kwargs
):
    data = {"instruction": instruction, "input": input, "output": None}
    prompt = generate_prompt_fn(data)["text"]

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=1.1,
        use_cache=True,
        early_stopping=True,
        do_sample=True,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", StoppingCriteriaList([stop_criteria])
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                yield prompter.get_response(decoded_output)
        return  # early return for stream_output

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    print(generation_output)
    s = generation_output.sequences[0]
    print(s)
    output = tokenizer.decode(s)
    print(output)

    yield output



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

    stop_criteria = StoppingCriteria()
    prompter = Prompter(data_args.prompt_template_name,
                        template_json_path=data_args.prompt_path,
                        is_chat_model=training_args.is_chat_model)
    formatting_prompts_func = partial(generate_prompt, prompter=prompter)

    model = PeftModel.from_pretrained(model, training_args.output_dir)
    evaluate_fn = partial(evaluate,
                          prompter=prompter,
                          device=device_map,
                          tokenizer=tokenizer,
                          generate_prompt_fn=formatting_prompts_func,
                          stop_criteria=stop_criteria,
                          model=model)
    demo(evaluate_fn=evaluate_fn, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
