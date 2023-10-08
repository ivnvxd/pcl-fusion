from collections import deque
from functools import reduce
from typing import Any, Dict, List, Optional

import numpy as np


class HostSync:
    """
    Synchronise messages from multiple streams based on their sequence numbers
    """

    def __init__(self, streams: List[str], maxlen: int = 50) -> None:
        self.queues = {stream: deque(maxlen=maxlen) for stream in streams}

    def add(self, stream: str, msg: Any) -> None:
        """
        Add a message to the queue for the given stream

        Args:
            stream: The stream to add the message to
            msg: The message to add
        """

        self.queues[stream].append({"msg": msg, "seq": msg.getSequenceNum()})

    def get(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest message from each stream, if they all have the same sequence number

        Returns:
            A dictionary of the latest message from each stream, or None if the sequence numbers don't match
        """  # noqa: E501

        seqs = [np.array([msg["seq"] for msg in msgs]) for msgs in self.queues.values()]
        matching_seqs = reduce(np.intersect1d, seqs)
        if len(matching_seqs) == 0:
            return None

        seq = np.max(matching_seqs)
        res = {
            stream: next(msg["msg"] for msg in msgs if msg["seq"] == seq)
            for stream, msgs in self.queues.items()
        }

        self.queues = {
            stream: deque([msg for msg in msgs if msg["seq"] > seq], maxlen=msgs.maxlen)
            for stream, msgs in self.queues.items()
        }

        return res
