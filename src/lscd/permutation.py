import numpy as np
import numpy.typing as npt

from src.lscd.model import GradedModel
from src.lemma import Lemma
from src.use import Use
from src.wic import ContextualEmbedder


class Permutation(GradedModel):
    wic: ContextualEmbedder
    n_perms: int
    whiten: bool
    k: int | None

    @staticmethod
    def compute_kernel_bias(
        vecs: npt.NDArray[np.float32], k: int | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        vecs = matrix (n x 768) with the sentence representations of your whole
        dataset (in the paper they use train, val and test sets)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        if k:
            return W[:, :k], -mu
        else:
            return W, -mu

    @staticmethod
    def transform_and_normalize(
        vecs: npt.NDArray[np.float32],
        kernel: npt.NDArray[np.float32] | None = None,
        bias: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Kernel and bias are W and -mu from previous function. They're passed to
        this function when inputing vecs vecs = vectors we need to whiten.
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return vecs / (vecs**2).sum(axis=1, keepdims=True) ** 0.5

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

    def predict(self, lemma: Lemma) -> float:
        earlier_df = lemma.uses_df[lemma.uses_df.grouping == lemma.groupings[0]]
        later_df = lemma.uses_df[lemma.uses_df.grouping == lemma.groupings[1]]

        earlier_uses = [Use.from_series(s) for _, s in earlier_df.iterrows()]
        later_uses = [Use.from_series(s) for _, s in later_df.iterrows()]

        with self.wic:
            earlier = np.vstack([self.wic.encode(use) for use in earlier_uses])
            later = np.vstack([self.wic.encode(use) for use in later_uses])

        observations = []
        first_observed = np.mean(
            self.euclidean_dist(earlier, later).flatten()
        )

        if self.whiten:
            kernel, bias = self.compute_kernel_bias(
                vecs=np.vstack([earlier, later]), 
                k=self.k
            )
            earlier = self.transform_and_normalize(earlier, kernel, bias)
            later= self.transform_and_normalize(later, kernel, bias)

        for _ in range(self.n_perms):
            perm_m0, perm_m1 = self.shuffle_matrices(m0=earlier, m1=later)
            distance = self.euclidean_dist(perm_m0, perm_m1)
            observations.append(np.mean(distance.flatten()))

        p_value = (
            len([obs for obs in observations if obs > first_observed]) / self.n_perms
        )
        return p_value
