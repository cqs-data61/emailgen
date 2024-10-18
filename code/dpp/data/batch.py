import torch

from typing import List, Optional

from dpp.utils import DotDict, pad_sequence
from .sequence import Sequence


class Batch(DotDict):
    """
    A batch consisting of padded sequences.

    Usually constructed using the from_list method.

    Args:
        inter_times: Padded inter-event times, shape (batch_size, seq_len)
        mask: Mask indicating which inter_times correspond to observed events
            (and not to padding), shape (batch_size, seq_len)
        marks: Padded marks associated with each event, shape (batch_size, seq_len)
    """
    def __init__(self, inter_times: torch.Tensor, mask: torch.Tensor, src_marks: Optional[torch.Tensor] = None, dst_marks: Optional[torch.Tensor] = None, meta: Optional[torch.Tensor] = None, pt_size: Optional[int] = 1, **kwargs):
        self.inter_times = inter_times
        self.mask = mask
        self.src_marks = src_marks
        self.dst_marks = dst_marks
        self.meta = meta
        self.pt_size = pt_size

        for key, value in kwargs.items():
            self[key] = value

        self._validate_args()

    @property
    def size(self):
        """Number of sequences in the batch."""
        return self.inter_times.shape[0]

    @property
    def max_seq_len(self):
        """Length of the padded sequences."""
        return self.inter_times.shape[1]

    def _validate_args(self):
        """Check if all tensors have correct shapes."""
        if self.inter_times.ndim != 2:
            raise ValueError(
                f"inter_times must be a 2-d tensor (got {self.inter_times.ndim}-d)"
            )
        if self.mask.shape != (self.size, self.max_seq_len):
            raise ValueError(
                f"mask must be of shape (batch_size={self.size}, "
                f" max_seq_len={self.max_seq_len}), got {self.mask.shape}"
            )
        if self.src_marks is not None and self.src_marks.shape != (self.size, self.max_seq_len):
            raise ValueError(
                f"source marks must be of shape (batch_size={self.size},"
                f" max_seq_len={self.max_seq_len}), got {self.src_marks.shape}"
            )
        if self.dst_marks is not None and self.dst_marks.shape != (self.size, self.max_seq_len):
            raise ValueError(
                f"destination marks must be of shape (batch_size={self.size},"
                f" max_seq_len={self.max_seq_len}), got {self.dst_marks.shape}"
            )
        if self.meta is not None and self.meta.shape != (self.size, self.max_seq_len, ):
            raise ValueError(
                f"meta must be of shape (batch_size={self.size},"
                f" max_seq_len={self.max_seq_len}), got {self.meta.shape}"
            )    
            
    @staticmethod
    def from_list(sequences: List[Sequence]):
        batch_size = len(sequences)
        # Remember that len(seq) = len(seq.inter_times) = len(seq.marks) + 1
        # since seq.inter_times also includes the survival time until t_end
        max_seq_len = max(len(seq) for seq in sequences)
        pt_size = 0
        for seq in sequences:
            pt_size += len(seq) -1
            
        inter_times = pad_sequence([seq.inter_times for seq in sequences], max_len=max_seq_len)

        dtype = sequences[0].inter_times.dtype
        device = sequences[0].inter_times.device
        mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=dtype)

        for i, seq in enumerate(sequences):
            mask[i, :len(seq) - 1] = 1

        if sequences[0].src_marks is not None:
            src_marks = pad_sequence([seq.src_marks for seq in sequences], max_len=max_seq_len)
        else:
            src_marks = None
            
        if sequences[0].dst_marks is not None:
            dst_marks = pad_sequence([seq.dst_marks for seq in sequences], max_len=max_seq_len)
        else:
            dst_marks = None
            
        if sequences[0].meta is not None:
            meta = pad_sequence([seq.meta for seq in sequences], max_len=max_seq_len)
        else:
            meta = None            

        return Batch(inter_times, mask, src_marks, dst_marks, meta, pt_size)

    def get_sequence(self, idx: int) -> Sequence:
        length = int(self.mask[idx].sum(-1)) + 1
        inter_times = self.inter_times[idx, :length]
        if self.src_marks is not None:
            src_marks = self.src_marks[idx, :length - 1, :]
        else:
            src_marks = None
        if self.dst_marks is not None:
            dst_marks = self.dst_marks[idx, :length - 1, :]
        else:
            dst_marks = None
        if self.meta is not None:
            meta = self.meta[idx, :length - 1]
        else:
            meta = None    
        # TODO: recover additional attributes (passed through kwargs) from the batch
        return Sequence(inter_times=inter_times, src_marks=src_marks, dst_marks=dst_marks, meta=meta)

    def to_list(self) -> List[Sequence]:
        """Convert a batch into a list of variable-length sequences."""
        return [self.get_sequence(idx) for idx in range(self.size)]
