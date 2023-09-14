from typing import Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    prompt_path: Optional[str] = field(
        default="/data/alpaca.json",
        metadata={"help": "Default prompt"},
    )
    prompt_template_name: Optional[str] = field(
        default="alpaca",
        metadata={"help": "Default prompt model nam"},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "LoRA alpha"}
    )
    lora_r: Optional[int] = field(
        default=64, metadata={"help": "LoRA r"}
    )
    lora_dropout: Optional[int] = field(
        default=0.1, metadata={"help": "LoRA dropout"}
    )
    use_4bit: Optional[bool] = field(
        default=True, metadata={"help": "Use 4 bit"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Bnb 4 bit quant type"}
    )
    compute_dtype: Optional[str] = field(
        default="float16", metadata={"help": "Torch dtype (float16, bfloat16)"}
    )
    use_nested_quant: Optional[bool] = field(
        default=False, metadata={"help": "BnB nested quant (for 4-bit based model)"}
    )
    max_seq_length: Optional[int] = field(
        default=None, metadata={"help": "Max training length"}
    )
    packing: Optional[bool] = field(
        default=False, metadata={"help": "Supervised finetuning dataset packing"}
    )
