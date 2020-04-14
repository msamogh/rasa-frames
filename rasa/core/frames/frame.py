import logging
from typing import Any, List, Text, Tuple, Dict, Optional, Callable


logger = logging.getLogger(__name__)


class Frame(object):
    """A dict-like wrapper for the slots inside a frame."""

    def __init__(
        self,
        idx: int,
        slots: Dict[Text, Any],
        created: float,
        switch_to: Optional[bool] = False,
    ) -> None:
        self.slots = slots
        self.idx = idx
        self.created = created
        self.last_active = created if switch_to else None

    def items(self) -> Dict[Text, Any]:
        return self.slots.items()

    def __getitem__(self, key: Text) -> Optional[Any]:
        return self.slots.get(key, None)

    def __setitem__(self, key: Text, value: Any) -> None:
        self.slots[key] = value


class FrameSet(object):
    """Container for all the frames in an agent's memory.

    It also keeps track of the current/active frame at any given point
    in time.

    The FrameSet of a tracker can be accessed using `tracker.frames`.
    """

    def __init__(self) -> None:
        """Initialize an empty FrameSet.

        The `current_frame_idx` is set to None."""
        self.frames = []
        self.current_frame_idx = None

    @property
    def current_frame(self) -> Frame:
        """Get the current frame.
        
        Returns None if the FrameSet is empty or there is no current frame.
        """
        if len(self.frames) == 0 or self.current_frame_idx is None:
            return None

        return self.frames[self.current_frame_idx]

    def add_frame(
        self, slots: Dict[Text, Any], created: float, switch_to: Optional[bool] = False
    ) -> Frame:
        """Insert a new frame into the FrameSlot.
        
        Args:
            slots: The slots to be inserted as key-value pairs.
            created: Timestamp of the time at which the new frame was created.
            switch_to: If True, makes the newly inserted Frame the current frame.

        Returns:
            Frame: The newly created frame.
        """
        logger.debug(f"Frame created with values {slots}")
        frame = Frame(
            idx=len(self.frames), slots=slots, created=created, switch_to=switch_to,
        )
        self.frames.append(frame)
        if switch_to:
            self.current_frame_idx = frame.idx
        return frame

    def __getitem__(self, idx: int) -> Frame:
        """Get the frame at index `idx`."""
        return self.frames[idx]

    def __len__(self) -> int:
        """Get the number of frames in the FrameSet."""
        return len(self.frames)

    def __str__(self) -> Text:
        """Display the contents of all frames in memory."""
        s = ""
        for idx, frame in enumerate(self.frames):
            s += f"\nFrame {idx}\n============ ({self.current_frame_idx})\n"
            s += str({key: slot for key, slot in frame.items()})
        return s

    def activate_frame(self, idx: int, timestamp: float) -> None:
        """Mark the frame at index `idx` as the current frame.
        
        Args:
            idx: Index of the frame to activate.
            timestamp: Timestamp .
        """
        self.current_frame_idx = idx
        self.frames[idx].last_active = timestamp

    @staticmethod
    def get_framed_entities(
        entities: Dict[Text, Any], domain: "Domain"
    ) -> Dict[Text, Any]:
        """Keep only those entities that correspond to `frame_slot`s."""
        framed_slot_names = [
            slot.name for slot in domain.slots if slot.frame_slot]
        framed_entities = {
            entity["entity"]: entity["value"]
            for entity in entities
            if entity["entity"] in framed_slot_names
        }
        return framed_entities
