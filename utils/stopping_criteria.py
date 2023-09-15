import torch
from transformers import StoppingCriteria


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids=None) -> None:
        super().__init__()
        self.stop_token_ids = stop_token_ids if stop_token_ids else list()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
