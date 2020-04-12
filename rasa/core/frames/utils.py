import copy
from typing import List, Dict, Text, Any, Callable

from rasa.core.events import Event, FrameCreated, FrameUpdated, \
    CurrentFrameDumped
from rasa.core.frames.frame_policy import FrameSet


def push_slots_into_current_frame(tracker: "DialogueStateTracker") -> List[Event]:
    events = []
    framed_entities = {
        key: slot.value
        for key, slot in tracker.slots.items()
        if slot.frame_slot
    }
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


def frames_from_tracker_slots(tracker: "DialogueStateTracker") -> FrameSet:
    """Temporarily generate the latest FrameSet until the tracker is permanently updated.
    
    Since the events from push_slots_into_current_frame would not have registered
    in the tracker yet by this point, create a temporary version of the FrameSet
    that reflects this change.
    """
    latest_tracker_slots = {
        key: slot.value
        for key, slot in tracker.slots.items()
        if slot.frame_slot
    }

    frames = copy.deepcopy(tracker.frames)
    if not frames.current_frame:
        frames.add_frame(
            slots=latest_tracker_slots,
            created=time.time(),
            switch_to=True
        )
    else:
        for key, value in latest_tracker_slots.items():
            frames.current_frame[key] = value

    return frames
