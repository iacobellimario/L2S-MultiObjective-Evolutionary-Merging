from logging import getLogger
import numpy as np
import pandas as pd
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.tasks import TaskManager
from .evaluator import LanguageDetector
from pprint import pprint

logger = getLogger(__name__)

class LmHarnessEvaluator:
    task: "ConfigurableTask"
    task_manager: TaskManager = None
    sample_ids: list[int] | None
    lang_detector: LanguageDetector | None
    lang_id: str | None

    def __init__(
        self,
        task_name: str,
        correctness_metric: str = "exact_match",
        sample_ids: list[int] | None = None,
        lang_id: str | None = None,
        is_test: bool = False,
        additional_templates_folder: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.task_manager = TaskManager(include_path=additional_templates_folder)
        self.task: "ConfigurableTask" = self.get_task(task_name)
        self.is_test = is_test
        self.sample_ids = sample_ids
        self.correctness_metric = correctness_metric
        self.task_nm = task_name
        self.lang_id = lang_id
        self.batch_size = batch_size

        if sample_ids is not None and len(sample_ids) > 0:
            dataset = self.task.dataset["test"]
            ids = np.arange(len(dataset))
            if is_test:
                self.task.dataset["test"] = dataset.select(np.setdiff1d(ids, sample_ids))
            else:
                self.task.dataset["test"] = dataset.select(sample_ids)
                logger.debug("Sample ids provided. Using the specified sample ids.")
        else:
            logger.info("No sample ids provided. Using the entire dataset.")

        if self.task.OUTPUT_TYPE == "multiple_choice":
            logger.warning("Disabling language detection for multiple choice tasks.")
            lang_id = None

        try:
            self.lang_detector = (
                LanguageDetector([lang_id]) if lang_id is not None else None
            )
        except Exception as e:
            logger.warning(f"Language detection is disabled. Error: {e}")
            self.lang_detector = None

    def get_task(self, task_name: str):
        return self.task_manager.load_task_or_group(task_name)[task_name]

    def evaluate(self, model) -> pd.Series:
        results = simple_evaluate(model, tasks=[self.task], batch_size=self.batch_size)
       
        if self.sample_ids is not None and not self.is_test:
            for sample in results["samples"][self.task_nm]:
                sample["doc_id"] = self.sample_ids[sample["doc_id"]]

        answers = []
        for sample in results["samples"][self.task_nm]:
            doc_id = sample["doc_id"]
            correctness = sample[self.correctness_metric]
            target = sample.get("target", "N/A")
            resps = sample["resps"]

            if isinstance(resps, list):
                if isinstance(resps[0], list):
                    model_ans = [item for sublist in resps for item in sublist]
                else:
                    model_ans = resps[0]
            else:
                model_ans = resps

            model_ans_str = (
                " ".join(map(str, model_ans)) if isinstance(model_ans, list) else str(model_ans)
            )

            answers.append({
                "id": doc_id,
                "correctness": correctness,
                "model_answers": model_ans_str,
                "target": target,
            })

        self.data = pd.DataFrame(answers)

        if self.sample_ids is not None:
            self.data = self.data[self.data["id"].isin(self.sample_ids)]
            if len(self.data) > len(self.sample_ids):
                self.data = self.data.groupby("id").sample(n=1).reset_index(drop=True)

        self.data["lang_correctness"] = 1.0
        if self.lang_detector is not None:
            self.data["language"] = self.data["model_answers"].apply(
                lambda x: self.lang_detector._get_language(x)
            )
            self.data["lang_correctness"] = (
                self.data["language"] == f"__label__{self.lang_id}"
            ).astype(float)

        self.data["answer_length"] = self.data["model_answers"].apply(lambda x: len(str(x).split()))

        self.data["fitness"] = (
            self.data["correctness"].astype(float) * 
            self.data["lang_correctness"].astype(float)
        )

        return self.data["fitness"]

    def get_data(self) -> pd.DataFrame:
        return self.data

def avg_lang_correctness(lang_id: str, langs: list[str]) -> float:
    if not langs:
        return np.nan
    return np.mean([lang == "UNK" or lang == f"__label__{lang_id}" for lang in langs])