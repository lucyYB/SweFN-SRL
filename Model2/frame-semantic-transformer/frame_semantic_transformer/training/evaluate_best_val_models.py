from __future__ import annotations

import os

from transformers import MT5ForConditionalGeneration, T5TokenizerFast

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.loaders.loader import (
    TrainingLoader,
)
from frame_semantic_transformer.training.evaluate_model import evaluate_model

from frame_semantic_transformer.training.find_best_val_model_paths import (
    find_best_val_model_paths,
)


def evaluate_best_val_models(
    outputs_dir: str,
    loader_cache: LoaderDataCache,
    training_loader: TrainingLoader,
    batch_size: int,
    num_workers: int,
) -> None:
    """
    Helper to run the evaluation on the best models after training
    """

    best_models = find_best_val_model_paths(outputs_dir)
    evaluated_models = set()
    for key, output_name in best_models.items():
        if output_name in evaluated_models:
            continue
        print(f"Best {key} model: {output_name}")
        model_path = os.path.join(outputs_dir, output_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5TokenizerFast.from_pretrained(model_path)
        print(f"Evaluating model: {output_name}")
        evaluate_model(
            model,
            tokenizer,
            loader_cache=loader_cache,
            training_loader=training_loader,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        evaluated_models.add(output_name)
