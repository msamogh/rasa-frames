import copy
import logging
import time
from collections import Counter
from typing import Any, List, Text, Tuple, Dict, Optional, Callable

from rasa.core.events import (
    Event,
    FrameCreated,
    CurrentFrameChanged,
    CurrentFrameDumped,
    FrameUpdated,
)
from rasa.core.utils import get_best_matching_frame_idx, is_first_frame_created_now
from rasa.core.frames import Frame, FrameSet


logger = logging.getLogger(__name__)


def push_slots_into_current_frame(tracker: "DialogueStateTracker") -> List[Event]:
    events = []
    framed_entities = {
        key: slot.value
        for key, slot in tracker.slots.items()
        if slot.frame_slot
    }
    logger.debug(
        f"Dumping slots into current frame... "
        f"{tracker.frames.current_frame} and {framed_entities}"
    )
    if tracker.frames.current_frame:
        for key, value in framed_entities.items():
            events.append(
                FrameUpdated(
                    frame_idx=tracker.frames.current_frame_idx,
                    name=key,
                    value=value,
                )
            )
    else:
        events.append(
            FrameCreated(
                slots=framed_entities,
                switch_to=True
            )
        )
    return events

def pop_slots_from_current_frame() -> List[Event]:
    return [CurrentFrameDumped()]