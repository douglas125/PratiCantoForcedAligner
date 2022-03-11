import numpy as np


class PFADecoder:
    def __init__(self):
        self.bigM = -1e6

    def decode_alignment(self, m):
        """Decodes character alignment

        Arguments:

        m - numpy array:
            score matrix with shape (n_chars, n_spectrograms)
        """
        # reset accumulated score
        self.accum_score = np.zeros_like(m) - 1.0
        self.accum_score[0, 0] = m[0, 0]
        self.decoded_path = np.zeros((*m.shape, 2), dtype=int)
        self._decode_accum_scores(m, m.shape[0] - 1, m.shape[1] - 1)
        best_path = self._decode_best_path(m)
        return best_path

    def _decode_best_path(self, m):
        cur_position = np.array([m.shape[0] - 1, m.shape[1] - 1], dtype=int)
        best_path = [[cur_position[0], cur_position[1]]]
        while cur_position.any() != 0:
            cur_position = self.decoded_path[(cur_position[0], cur_position[1])]
            best_path.append([cur_position[0], cur_position[1]])
        best_path.reverse()
        return best_path

    def _decode_accum_scores(self, m, i, j):
        if i == 0 and j == 0:
            return m[0, 0]
        elif i < 0 or j < 0:
            return self.bigM

        if self.accum_score[i, j] < 0:
            same_row_score = self._decode_accum_scores(m, i, j - 1) + m[i, j]
            switch_row_score = self._decode_accum_scores(m, i - 1, j - 1) + m[i, j]
            if same_row_score > switch_row_score:
                self.accum_score[i, j] = same_row_score
                self.decoded_path[i, j] = [i, j - 1]
            else:
                self.accum_score[i, j] = switch_row_score
                self.decoded_path[i, j] = [i - 1, j - 1]
        return self.accum_score[i, j]
