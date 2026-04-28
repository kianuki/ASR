import re

import sentencepiece as spm
import torch

from src.utils.io_utils import ROOT_PATH
from src.utils.tokenizer_utils import prepare_transcriptions


class BPEEncoder:
    EMPTY_TOK = "<blank>"

    def __init__(
        self,
        vocab_size=1000,
        needs_training=False,
        dataset=None,
        partition=None,
        **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.sp = None
        self.vocab_size = vocab_size

        self.needs_training = needs_training
        if self.needs_training:
            self._train(dataset, partition)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(ROOT_PATH / "src" / "text_encoder" / "m_bpe.model"))

        self.EMPTY_IND = self.sp.piece_to_id(self.EMPTY_TOK)

    def __len__(self):
        return self.sp.get_piece_size()

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.sp.id_to_piece(item)

    def _train(self, dataset, partition):
        path = prepare_transcriptions(dataset=dataset, partition=partition)
        spm.SentencePieceTrainer.train(
            input=path,
            model_prefix="m_bpe",
            vocab_size=self.vocab_size,
            user_defined_symbols=[self.EMPTY_TOK],
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
        )

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.tensor(self.sp.encode_as_ids(text), dtype=torch.long)
        except KeyError:
            raise Exception(f"Can't encode text '{text}'")

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        if hasattr(inds, "tolist"):
            inds = inds.tolist()

        return self.sp.decode_ids(inds)

    def ctc_decode(self, inds) -> str:
        if hasattr(inds, "tolist"):
            inds = inds.tolist()

        prev_ind = None
        ctc_inds = []

        for ind in inds:
            cur_ind = int(ind)

            if cur_ind == prev_ind or cur_ind == self.EMPTY_IND:
                continue

            ctc_inds.append(cur_ind)
            prev_ind = cur_ind

        return self.sp.decode_ids(ctc_inds)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
