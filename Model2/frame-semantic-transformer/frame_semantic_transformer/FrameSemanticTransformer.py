from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, cast
import torch
from transformers import MT5ForConditionalGeneration, T5TokenizerFast

from frame_semantic_transformer.constants import (
    MODEL_MAX_LENGTH,
    MODEL_REVISION,
    OFFICIAL_RELEASES,
)
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.data_utils import chunk_list, marked_string_to_locs
from frame_semantic_transformer.data.loaders.loader import InferenceLoader
from frame_semantic_transformer.data.loaders.framenet17 import Framenet17InferenceLoader
from frame_semantic_transformer.predict import batch_predict
from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionTask,
    FrameClassificationTask,
    TriggerIdentificationTask,
)


@dataclass
class FrameElementResult:
    name: str
    text: str


@dataclass
class FrameResult:
    name: str
    trigger_location: int
    frame_elements: list[FrameElementResult]


@dataclass
class DetectFramesResult:
    sentence: str
    trigger_locations: list[int]
    frames: list[FrameResult]


class FrameSemanticTransformer:
    _model: MT5ForConditionalGeneration | None = None
    _tokenizer: T5TokenizerFast | None = None
    model_path: str
    model_revision: str | None = None
    device: torch.device
    max_batch_size: int
    predictions_per_sample: int
    loader_cache: LoaderDataCache

    def __init__(
        self,
        model_name_or_path: str = "base",
        use_gpu: bool = torch.cuda.is_available(),
        max_batch_size: int = 8,
        predictions_per_sample: int = 5,
        inference_loader: Optional[InferenceLoader] = None,
    ):
        self.model_path = model_name_or_path
        if model_name_or_path in OFFICIAL_RELEASES:
            self.model_path = f"chanind/frame-semantic-transformer-{model_name_or_path}"
            self.model_revision = MODEL_REVISION
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.max_batch_size = max_batch_size
        self.predictions_per_sample = predictions_per_sample
        loader = inference_loader or Framenet17InferenceLoader()
        self.loader_cache = LoaderDataCache(loader)

    def setup(self) -> None:
        """
        Initialize the model and tokenizer, and download models / files as needed
        If this is not called explicitly it will be lazily called before inference
        """
        self._model = MT5ForConditionalGeneration.from_pretrained(
            self.model_path, revision=self.model_revision
        ).to(self.device)
        self._tokenizer = T5TokenizerFast.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            model_max_length=MODEL_MAX_LENGTH,
        )
        self.loader_cache.setup()
        self._validate_loader()

    def _validate_loader(self) -> None:
        """
        Helper to ensure that the loader being used matches the one used to train the model
        otherwise results will be potentially nonsensical
        """
        loader = self.loader_cache.loader
        config = self.model.config
        expected_loader_name = (
            config.inference_loader
            if hasattr(config, "inference_loader")
            else loader.name()
        )
        if expected_loader_name != loader.name():
            raise ValueError(
                f"Model {self.model_path} was trained with inference loader {expected_loader_name} "
                f"but you are using {loader.name()}. Please pass in the correct loader as 'inference_loader' "
                f"when initializing FrameSemanticTransfomer."
            )

    @property
    def model(self) -> MT5ForConditionalGeneration:
        if not self._model:
            self.setup()
        return cast(MT5ForConditionalGeneration, self._model)

    @property
    def tokenizer(self) -> T5TokenizerFast:
        if not self._tokenizer:
            self.setup()
        return cast(T5TokenizerFast, self._tokenizer)

    def _batch_predict(self, inputs: list[str]) -> list[str]:
        """
        helper to avoid needing to repeatedly pass in the same params every call to predict
        """
        return batch_predict(
            self.model,
            self.tokenizer,
            inputs,
            num_beams=self.predictions_per_sample,
            num_return_sequences=self.predictions_per_sample,
        )

    def _identify_triggers(self, sentence: str) -> tuple[str, list[int]]:
        task = TriggerIdentificationTask(text=sentence)
        outputs = self._batch_predict([task.get_input()])
        result = task.parse_output(outputs, self.loader_cache)
        return marked_string_to_locs(result)

    def _classify_frames(
        self, sentence: str, trigger_locs: list[int]
    ) -> list[str | None]:
        """
        Return a list containing a frame for each trigger_loc passed in.
        If no frame can be found, None is returned for the frame instead.
        """
        frame_classification_tasks: list[FrameClassificationTask] = []
        frames: list[str | None] = []

        for trigger_loc in trigger_locs:
            frame_classification_tasks.append(
                FrameClassificationTask(
                    text=sentence,
                    trigger_loc=trigger_loc,
                    loader_cache=self.loader_cache,
                )
            )
        for batch in chunk_list(
            frame_classification_tasks, chunk_size=self.max_batch_size
        ):
            batch_results = self._batch_predict([task.get_input() for task in batch])
            for preds, frame_task in zip(
                chunk_list(batch_results, self.predictions_per_sample),
                batch,
            ):
                frames.append(frame_task.parse_output(preds, self.loader_cache))
        return frames

    def _extract_frame_args(
        self, sentence: str, frame_with_trigger_locs: list[tuple[str, int]]
    ) -> list[list[tuple[str, str]]]:
        """
        return a list of tuples of (frame_element, text) for each frame/trigger loc passed in.
        The returned list will have the same length as the frame_with_trigger_locs list,
        with each element corresponding to a frame/loc in the input list
        """
        frame_element_results: list[list[tuple[str, str]]] = []
        arg_extraction_tasks = [
            ArgumentsExtractionTask(
                text=sentence,
                trigger_loc=trigger_loc,
                frame=frame,
                loader_cache=self.loader_cache,
            )
            for frame, trigger_loc in frame_with_trigger_locs
        ]
        for args_tasks_batch in chunk_list(
            arg_extraction_tasks, chunk_size=self.max_batch_size
        ):
            batch_results = self._batch_predict(
                [task.get_input() for task in args_tasks_batch],
            )
            for preds, args_task in zip(
                chunk_list(batch_results, self.predictions_per_sample), args_tasks_batch
            ):
                frame_element_results.append(
                    args_task.parse_output(preds, self.loader_cache)
                )
        return frame_element_results

    def detect_frames(self, sentence: str) -> DetectFramesResult:
        # first detect trigger locations
        base_sentence, trigger_locs = self._identify_triggers(sentence)
        # next detect frames for each trigger
        frames = self._classify_frames(base_sentence, trigger_locs)

        frame_and_locs = [
            (frame, loc) for frame, loc in zip(frames, trigger_locs) if frame
        ]
        frame_elements_lists = self._extract_frame_args(base_sentence, frame_and_locs)
        frame_results: list[FrameResult] = []
        for ((frame, loc), frame_element_tuples) in zip(
            frame_and_locs, frame_elements_lists
        ):
            frame_elements = [
                FrameElementResult(element, text)
                for element, text in frame_element_tuples
            ]
            frame_results.append(
                FrameResult(
                    name=frame,
                    trigger_location=loc,
                    frame_elements=frame_elements,
                )
            )
        return DetectFramesResult(
            base_sentence,
            trigger_locations=trigger_locs,
            frames=frame_results,
        )
