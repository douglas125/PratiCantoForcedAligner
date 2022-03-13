import numpy as np


class PFADecoder:
    def __init__(self):
        self.bigM = -1e6

    def greedy_decode(self, m):
        """Greeedy decoder of character alignment

        Arguments:

        m - numpy array:
            score matrix with shape (n_chars, n_spectrograms)
        """
        greedy_assignment = np.argmax(m, axis=0)
        cur_max = greedy_assignment[0]
        for idx in range(len(greedy_assignment)):
            cur_max = max(cur_max, greedy_assignment[idx])
            if greedy_assignment[idx] < cur_max:
                greedy_assignment[idx] = cur_max
        return greedy_assignment

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
        return [x[0] for x in best_path]

    def _decode_accum_scores(self, m, i, j):
        if i == 0 and j == 0:
            return m[0, 0]
        elif i < 0 or j < 0:
            return self.bigM

        if self.accum_score[i, j] == -1.0:
            same_row_score = self._decode_accum_scores(m, i, j - 1) + m[i, j]
            switch_row_score = self._decode_accum_scores(m, i - 1, j - 1) + m[i, j]
            if same_row_score > switch_row_score:
                self.accum_score[i, j] = same_row_score
                self.decoded_path[i, j] = [i, j - 1]
            else:
                self.accum_score[i, j] = switch_row_score
                self.decoded_path[i, j] = [i - 1, j - 1]
        return self.accum_score[i, j]

    # Static methods
    def write_srt(chars, alignment, time_delta, filename):
        char_dict = PFADecoder._compute_chardict(chars, alignment, time_delta, filename)
        PFADecoder._write_chardict(char_dict, filename)

    def _write_chardict(char_dict, filename):
        cur_annot = 1
        with open(filename, "w") as f:
            use_sep = False
            for x in char_dict:
                xmin = char_dict[x]["t0"]
                xmax = char_dict[x]["tf"]
                txt = char_dict[x]["char"]
                if use_sep:
                    f.write("\n\n")
                use_sep = True

                f.write(f"{cur_annot}\n")
                # don't show in UI
                base_txt = txt + "|||8760|||9760|||Arial, 44pt|||False|||"
                xmin = PFADecoder.convert_seconds_to_srt(xmin)
                xmax = PFADecoder.convert_seconds_to_srt(xmax)
                f.write(f"{xmin} --> {xmax}\n")
                f.write(f"{base_txt}")
                cur_annot += 1

    def _compute_chardict(chars, alignment, time_delta, filename):
        char_dict = {}
        prev_char = alignment[0]
        char_dict[prev_char] = {"char": chars[prev_char], "t0": 0}
        for spec_idx, char_idx in enumerate(alignment):
            if char_idx > prev_char:
                char_dict[prev_char]["tf"] = spec_idx * time_delta
                char_dict[char_idx] = {
                    "char": chars[char_idx],
                    "t0": spec_idx * time_delta,
                }
            prev_char = char_idx
        char_dict[prev_char]["tf"] = (len(alignment) - 1) * time_delta
        return char_dict

    def convert_seconds_to_srt(time_in_s):
        # 00:00:01,417 --> 00:00:01,924
        hours = int(time_in_s) // 3600
        remaining = int(time_in_s) - 3600 * hours
        minutes = remaining // 60
        seconds = remaining - 60 * minutes

        milliseconds = str(int(np.round(1000 * (time_in_s - int(time_in_s)))))

        hours = str(hours).rjust(2, "0")
        minutes = str(minutes).rjust(2, "0")
        seconds = str(seconds).rjust(2, "0")
        milliseconds = milliseconds.rjust(3, "0")
        return f"{hours}:{minutes}:{seconds},{milliseconds}"
