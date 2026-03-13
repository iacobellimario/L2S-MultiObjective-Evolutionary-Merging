from logging import getLogger
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from mergenetic.optimization import MergingProblem

logger = getLogger(__name__)

class Searcher:
    def __init__(
        self,
        problem: MergingProblem,
        algorithm: Algorithm,
        results_path: str,
        n_iter: int,
        seed: int,
        run_id: str,
        verbose: bool = True,
    ):
        self.problem = problem
        self.algorithm = algorithm
        self.results_path = Path(results_path)
        self.n_iter = n_iter
        self.seed = seed
        self.run_id = run_id
        self.verbose = verbose

    def search(self) -> pd.DataFrame | None:
        termination = get_termination("n_gen", self.n_iter)
        result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            termination=termination,
            seed=self.seed,
            verbose=self.verbose,
        )

        # Extract results
        self.result_X = result.X / 10 if getattr(self.problem, "discrete", False) else result.X
        self.result_F = result.F

        F = np.asarray(self.result_F)
        X = np.asarray(self.result_X)

        # Normalizza a 2D
        if F.ndim == 0:
            F = F.reshape(1, 1)
        elif F.ndim == 1:
            F = F.reshape(-1, 1)

        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(1, -1)  
            
        print(F.shape, X.shape)

        obj_cols  = [f"objective_{i}" for i in range(F.shape[1])]
        geno_cols = [f"genotype_{i}"  for i in range(X.shape[1])]

        F_df = pd.DataFrame(F, columns=obj_cols)
        X_df = pd.DataFrame(X, columns=geno_cols)

        combined = pd.concat([F_df.reset_index(drop=True), X_df.reset_index(drop=True)], axis=1)
        
        self.results_path.mkdir(parents=True, exist_ok=True)
        solutions_csv = self.results_path / f"{self.run_id}_solutions.csv"
        combined.to_csv(solutions_csv, index=False)

        logger.info(f"Solutions CSV saved: {solutions_csv}")

        # Export results_df if present
        if not hasattr(self.problem, "results_df"):
            logger.warning("No results_df attribute found in problem.")
            return None

        if isinstance(self.problem.results_df, pd.DataFrame):
            out = self.results_path / f"{self.run_id}.csv"
            self.problem.results_df.to_csv(out, index=False)
            return self.problem.results_df

        if isinstance(self.problem.results_df, dict):
            for key, df in self.problem.results_df.items():
                df.to_csv(self.results_path / f"{self.run_id}_{key}.csv", index=False)
            return self.problem.results_df

        logger.error("problem.results_df must be a DataFrame or a dict of DataFrames.")
        return None
    

    # DO NOT CALL test() and visualize_results()
    def test(self):
        """
        Evaluate best solution(s) on the test set and save fitness + genotypes only.
        """
        logger.info("Starting final evaluation on test set.")

        if not hasattr(self, "result_X") or not hasattr(self, "result_F"):
            logger.error("Missing result_X or result_F. You must run search() before test().")
            return

        # Ensure arrays are 2D
        X = self.result_X if self.result_X.ndim > 1 else np.array([self.result_X])
        F = self.result_F if self.result_F.ndim > 1 else np.array([self.result_F])

        test_records = []

        for i, genotype in enumerate(X):
            assert isinstance(genotype, np.ndarray), "Genotype is not a NumPy array."

            fitness, _ = self.problem.test(genotype)
            logger.info(f"[{i+1}] Genotype {genotype} got fitness {fitness}")

            if not isinstance(fitness, (list, np.ndarray)):
                fitness = [fitness]

            row = {f"objective_{i+1}": val for i, val in enumerate(fitness)}
            row.update({f"genotype_{i+1}": val for i, val in enumerate(genotype.tolist())})

            test_records.append(row)

        df = pd.DataFrame(test_records)
        df.to_csv(self.results_path / f"{self.run_id}_test.csv", index=False)

        logger.info("Test evaluation completed.")
    

    def visualize_results(self) -> None:
        """Plot optimization metrics and phenotypes from results_df."""
        if not hasattr(self.problem, "results_df"):
            raise AttributeError("Problem does not have 'results_df' to visualize.")

        df = self.problem.results_df

        if isinstance(df, dict):
            for key, sub_df in df.items():
                self._plot_metrics(sub_df, title_prefix=key)
        else:
            self._plot_metrics(df)

    def _plot_metrics(self, df: pd.DataFrame, title_prefix: str = ""):
        metrics = [col for col in df.columns if "objective" in col]
        phenotypes = [col for col in df.columns if "phenotype" in col]

        for metric in metrics:
            plt.figure(figsize=(10, 4))
            plt.plot(df["step"], df[metric], marker="o")
            plt.title(f" Metric: {title_prefix}")
            plt.xlabel("Step")
            plt.ylabel(metric)
            plt.grid(True)
            plt.show()

        for phenotype in phenotypes:
            plt.figure(figsize=(10, 4))
            plt.plot(df["step"], df[phenotype], marker="x", linestyle="--")
            plt.title(f"{title_prefix} Phenotype: {phenotype}")
            plt.xlabel("Step")
            plt.ylabel(phenotype)
            plt.grid(True)
            plt.show()
            
            