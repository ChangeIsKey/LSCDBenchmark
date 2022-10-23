import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.lscd.model import GradedModel
from src.target import Target
from src.use import Use
from src.wic import ContextualEmbedder


class Permutation(GradedModel):
    wic: ContextualEmbedder
    n_perms: int

    @staticmethod
    def euclidean_dist(
        m0: npt.NDArray[np.float32], m1: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        return (
            -2 * np.dot(m0, m1.T)
            + np.sum(m1**2, axis=1)
            + np.sum(m0**2, axis=1)[:, np.newaxis]
        )

    def get_n_rows(
        self, m0: npt.NDArray[np.float32], m1: npt.NDArray[np.float32]
    ) -> int:
        if m0.shape[0] <= m1.shape[0]:
            return np.random.randint(1, m0.shape[0])
        else:
            return np.random.randint(1, m1.shape[0])

    def shuffle_matrices(
        self,
        m0: npt.NDArray[np.float32],
        m1: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        n_rows = self.get_n_rows(m0, m1)
        random_indices = (
            sorted(np.random.choice(m0.shape[0], size=n_rows, replace=False)),
            sorted(np.random.choice(m1.shape[0], size=n_rows, replace=False)),
        )

        indices_to_keep = (
            [i for i in range(m0.shape[0]) if i not in random_indices[0]],
            [i for i in range(m1.shape[0]) if i not in random_indices[1]],
        )

        perm_m0 = np.zeros_like(m0)
        perm_m1 = np.zeros_like(m1)

        perm_m0[random_indices[0]] = m1[random_indices[1]]
        perm_m1[random_indices[1]] = m0[random_indices[0]]

        perm_m0[indices_to_keep[0]] = m0[indices_to_keep[0]]
        perm_m1[indices_to_keep[1]] = m1[indices_to_keep[1]]

        return perm_m0, perm_m1

    def predict(self, targets: list[Target]) -> dict[str, float]:
        predictions = {}
        earlier = []
        later = []
        for target in tqdm(targets, desc="Generating target predictions"):
            earlier_df = target.uses_df[target.uses_df.grouping == target.groupings[0]]
            later_df = target.uses_df[target.uses_df.grouping == target.groupings[1]]

            earlier_uses = [Use.from_series(s) for _, s in earlier_df.iterrows()]
            later_uses = [Use.from_series(s) for _, s in later_df.iterrows()]

            with self.wic:
                earlier.extend([self.wic.encode(use) for use in earlier_uses])
                later.extend([self.wic.encode(use) for use in later_uses])

            earlier_stacked = np.vstack(earlier)
            later_stacked = np.vstack(later)

            observations = []
            first_observed = np.mean(self.euclidean_dist(earlier_stacked, later_stacked).flatten())

            for _ in range(self.n_perms):
                perm_m0, perm_m1 = self.shuffle_matrices(earlier_stacked, later_stacked)
                distance = self.euclidean_dist(perm_m0, perm_m1)
                observations.append(np.mean(distance.flatten()))

            p_value = len([i for i in observations if i > first_observed]) / self.n_perms
            predictions[target.name] = p_value

        return predictions
