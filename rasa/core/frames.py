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
        framed_slot_names = [slot.name for slot in domain.slots if slot.frame_slot]
        framed_entities = {
            entity["entity"]: entity["value"]
            for entity in entities
            if entity["entity"] in framed_slot_names
        }
        return framed_entities


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


class RuleBasedFrameTracker(object):
    def __init__(self, domain: "Domain") -> None:
        self.domain = domain

    def predict(
        self, tracker: "DialogueStateTracker", user_utterance: "UserUttered"
    ) -> List[Event]:
        # 1. Update the current frame with slots that might have been updated
        # through other custom actions.
        # 2. Check for changes in either the slot-values within the current frame
        # or for a switching of the active frame.
        # 3. Refill the slots with the contents of the current frame
        events = push_slots_into_current_frame(tracker) + \
                    self._handle_frame_changes(tracker, user_utterance) + \
                    pop_slots_from_current_frame()

        return events

    def _get_latest_frames(self, tracker: "DialogueStateTracker") -> FrameSet:
        """Generate the latest FrameSet temporarily until the tracker is permanently updated."""
        latest_tracker_slots = {
            key: slot.value
            for key, slot in tracker.slots.items()
            if slot.frame_slot
        }

        frames = copy.deepcopy(tracker.frames)
        if not frames.current_frame:
            frames.add_frame(
                slots=latest_tracker_slots,
                timestamp=time.time(),
                switch_to=True
            )
        else:
            for key, value in latest_tracker_slots.items():
                frames.current_frame[key] = value

        return frames

    def _handle_frame_changes(
        self,
        tracker: "DialogueStateTracker",
        user_utterance: "UserUttered",
    ) -> List[Event]:
        def get_intent_frame_props(intent):
            can_contain_frame_ref = self.domain.intent_properties[intent]["can_contain_frame_ref"]
            on_frame_match_failed = self.domain.intent_properties[intent]["on_frame_match_failed"]
            return can_contain_frame_ref, on_frame_match_failed

        intent = user_utterance.intent
        can_contain_frame_ref, on_frame_match_failed = get_intent_frame_props(intent)
        dialogue_entities = FrameSet.get_framed_entities(
            user_utterance.entities, self.domain
        )

        if can_contain_frame_ref:
            events = self._update_or_switch_frame(
                tracker,
                on_frame_match_failed,
                dialogue_entities,
            )
            return events

        return []

    def _get_choice_fn(self, on_frame_match_failed: Text) -> Callable:

        def most_recent_frame_idx(frames: List[Frame]) -> int:
            most_recent_frame = list(
                sorted(frames, key=lambda frame: frame.created, reverse=True)
            )[0]
            return most_recent_frame.idx

        def create_new(
            matching_candidates: List[Frame],
            non_conflicting_candidates: List[Frame],
            all_frames: List[Frame]
        ) -> int:
            if matching_candidates:
                return most_recent_frame_idx(matching_candidates)
            if non_conflicting_candidates:
                if all_frames.current_frame_idx in non_conflicting_candidates:
                    return all_frames.current_frame_idx
            return len(all_frames)

        def most_recent(
            matching_candidates: List[Frame],
            non_conflicting_candidates: List[Frame],
            all_frames: List[Frame]
        ) -> int:
            if matching_candidates:
                return most_recent_frame_idx(matching_candidates)
            return most_recent_frame_idx(all_frames)

        if on_frame_match_failed == 'create_new':
            return create_new
        elif on_frame_match_failed == 'most_recent':
            return most_recent

    def _update_or_switch_frame(
        self,
        tracker: "DialogueStateTracker",
        on_frame_match_failed: Text,
        framed_entities: Dict[Text, Any]
    ) -> None:
        ref_frame_idx = get_best_matching_frame_idx(
            frames=self._get_latest_frames(tracker),
            framed_entities=framed_entities,
            choice_fn=self._get_choice_fn(on_frame_match_failed)
        )

        if ref_frame_idx == len(tracker.frames):
            return [FrameCreated(
                slots=framed_entities,
                switch_to=True
            )]

        return []
        