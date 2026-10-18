import math
import random
from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if not torch.isfinite(batch["loss"]):
            raise FloatingPointError("Non-finite loss")

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            if not math.isfinite(self._get_grad_norm()):
                raise FloatingPointError("Non-finite gradient norm")
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train", examples_to_log=10):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
            examples_to_log (int): maximum number of examples to log.
        """
        indices = random.sample(
            range(len(batch["text"])), min(examples_to_log, len(batch["text"]))
        )
        self.log_predictions(indices=indices, **batch)
        self.log_spectrogram(indices=indices, **batch)
        self.log_audio(indices=indices, sample_rate=self.sample_rates[mode], **batch)

    def log_spectrogram(
        self, spectrogram, spectrogram_length, audio_path, indices, **batch
    ):
        for i in indices:
            length = int(spectrogram_length[i])
            spectrogram_for_plot = spectrogram[i, :, :length].detach().cpu()
            image = plot_spectrogram(spectrogram_for_plot)
            self.writer.add_image(f"spectrogram_{Path(audio_path[i]).stem}", image)

    def log_audio(self, audio, audio_length, audio_path, indices, sample_rate, **batch):
        for i in indices:
            length = int(audio_length[i])
            self.writer.add_audio(
                f"audio_{Path(audio_path[i]).stem}",
                audio[i, :length],
                sample_rate=sample_rate,
            )

    def log_predictions(
        self, text, logits, logits_length, audio_path, indices, **batch
    ):
        rows = {}
        for i in indices:
            length = int(logits_length[i])
            sample_logits = logits[i, :length].detach().cpu()
            raw_pred = self.text_encoder.decode(sample_logits)
            pred = self.text_encoder.ctc_decode(sample_logits)
            target = self.text_encoder.normalize_text(text[i])
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            path = Path(audio_path[i])

            self.writer.add_text(
                f"prediction_{path.stem}",
                f"target: {target}, prediction: {pred}, CER: {cer:.2f}%, WER: {wer:.2f}%",
            )
            rows[path.name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
