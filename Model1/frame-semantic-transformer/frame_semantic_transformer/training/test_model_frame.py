from __future__ import annotations
import logging
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.TaskSampleDataset import TaskSampleDataset
# add for SweSRL
from frame_semantic_transformer.data.loaders.loader import TrainingLoader, InferenceLoader
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)
from .TrainingModelWrapper import TrainingModelWrapper

logger = logging.getLogger(__name__)

def evaluate_model(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    loader_cache: LoaderDataCache,
    training_loader: TrainingLoader,
    batch_size: int,
    num_workers: int,
    use_gpu: bool = torch.cuda.is_available(),
) -> None:
    """
    Evaluate this model against the test sets per Frame
    """
    logger.info("loading the chosen model")

    inference_loader = loader_cache.loader
    inference_loader.setup()
    training_loader.setup()

    model.config.training_loader = training_loader.name()
    model.config.inference_loader = inference_loader.name()
    loader_cache = LoaderDataCache(inference_loader)

    model_wrapper = TrainingModelWrapper(model, tokenizer, loader_cache)

    trainer = Trainer(
        accelerator='gpu', devices=1 if use_gpu else 0,
        # gpus = None,
        log_every_n_steps=10,
        precision=32, 
        max_epochs=1)
    
    logger.info("loading test data")
    test_dataset = TaskSampleDataset(
        tasks_from_annotated_sentences(training_loader.load_test_data(), loader_cache),
        tokenizer,
        balance_tasks=False,
    )

    with torch.no_grad():
        print("Evaluating on test set")
        results = trainer.test(model_wrapper,
                               dataloaders=DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers),
        )
    
    return results
