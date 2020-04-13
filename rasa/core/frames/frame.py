import logging
from typing import Any, List, Text, Tuple, Dict, Optional, Callable


logger = logging.getLogger(__name__)


class Frame(object):
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
    def __init__(self) -> None:
        self.frames = []
        self.current_frame_idx = None

    @property
    def current_frame(self) -> Frame:
        if len(self.frames) == 0 or self.current_frame_idx is None:
            return None

        return self.frames[self.current_frame_idx]

    def reset(self) -> None:
        logger.debug("!!!@@@Frames reset@@@!!!")
        self.frames = []
        self.current_frame_idx = None

    def add_frame(
        self, slots: Dict[Text, Any], created: float, switch_to: Optional[bool] = False
    ) -> Frame:
        logger.debug(f"Frame created with values {slots}")
        frame = Frame(
            idx=len(self.frames), slots=slots, created=created, switch_to=switch_to,
        )
        self.frames.append(frame)
        if switch_to:
            self.current_frame_idx = frame.idx
        return frame

    def __getitem__(self, idx: int) -> Frame:
        return self.frames[idx]

    def __len__(self) -> int:
        return len(self.frames)

    def __str__(self) -> Text:
        s = ""
        for idx, frame in enumerate(self.frames):
            s += f"\nFrame {idx}\n============\n"
            s += str({key: slot for key, slot in frame.items()})
        return s

    def activate_frame(self, idx: int, timestamp: float) -> None:
        self.current_frame_idx = idx
        self.frames[idx].last_active = timestamp

    @staticmethod
    def get_framed_entities(
        entities: Dict[Text, Any], domain: "Domain"
    ) -> Dict[Text, Any]:
        logger.debug(f"All those slots: {domain.slots}")
        logger.debug(entities)
        framed_slot_names = [slot.name for slot in domain.slots if slot.frame_slot]
        framed_entities = {
            entity["entity"]: entity["value"]
            for entity in entities
            if entity["entity"] in framed_slot_names
        }
        return framed_entities
