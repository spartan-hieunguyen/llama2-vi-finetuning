import torch
import gradio as gr
from transformers import GenerationConfig, StoppingCriteriaList


def print_hello():
    print("hello")
    b = c + a / 0
    return a / 0


def demo(evaluate_fn, server_name):
    gr.Interface(
        fn=evaluate_fn,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Hôm nay thời tiết như nào ?",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=10, value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Vietcuna",
        description="Vietcuna Test",  # noqa: E501
    ).queue().launch(server_name=server_name)
