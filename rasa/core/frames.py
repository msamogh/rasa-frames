import logging
from collections import Counter
from typing import Any, List, Text, Tuple, Dict, Optional

from rasa.core.events import Event, SlotSet, UserUttered
from rasa.core.events import (
    FrameCreated,
    CurrentFrameChanged,
    CurrentFrameDumped,
    FrameUpdated,
)
from rasa.core.slots import Slot


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
    def __init__(self, init_slots: Dict[Text, Any], created: float) -> None:
        self.frames = []
        self.current_frame_idx = None
        if init_slots:
            init_frame = Frame(idx=0, slots=init_slots, created=created, switch_to=True)
            self.frames.append(init_frame)

    @property
    def current_frame(self) -> Frame:
        if len(self.frames) == 0 or self.current_frame_idx is None:
            return None
        return self.frames[self.current_frame_idx]

    def reset(self) -> None:
        lgoger.debug("!!!@@@Frames reset@@@!!!")
        self.frames = []
        self.current_frame_idx = None

    def add_frame(
        self, slots: List[Slot], created: float, switch_to: Optional[bool] = False
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
            s += str({key: slot for key, slot in frame.slots.items()})
        return s

    def activate_frame(self, idx: int, timestamp: float) -> None:
        self.current_frame_idx = idx
        self.frames[idx].last_active = timestamp

    @staticmethod
    def get_framed_slots(entities: Dict[Text, Any], domain: "Domain") -> Dict[Text, Any]:
        return {slot.name: slot.value for slot in domain.slots if slot.frame_slot}


class RuleBasedFrameTracker(object):
    def __init__(domain: "Domain") -> None:
        self.domain = domain

    def update_frames(
        self, tracker: "DialogueStateTracker", user_utterance: Text
    ) -> List[Event]:
        # Treat the slot values in the tracker as temporary values
        # (not necessarily reflecting the values of the active frame).
        # The active frame will be decided upon only after checking with the FrameTracker.
        dialogue_act = user_utterance.intent["name"]
        dialogue_entities = FrameSet.get_framed_slots(
            user_utterance.entities, self.domain
        )

        acts_with_ref = [
            "affirm",
            "canthelp",
            "confirm",
            "hearmore",
            "inform",
            "moreinfo",
            "negate",
            "no_result",
            "offer",
            "request",
            "request_compare",
            "suggest",
            "switch_frame",
        ]

        events = []
        if dialogue_act == "inform":
            logger.debug("Handling inform")
            events.extend(self.handle_inform(tracker, dialogue_entities))
        elif dialogue_act == "switch_frame":
            logger.debug("Switching frame")
            events.extend(self.handle_switch_frame(tracker, dialogue_entities))
        elif dialogue_act in acts_with_ref:
            # anything with a ref tag
            logger.debug("Handling act with ref")
            events.extend(self.handle_act_with_ref(tracker, dialogue_entities))
        else:
            logger.debug("Handling another kind of intent")
            for key, value in dialogue_entities.items():
                events.append(
                    FrameUpdated(
                        frame_idx=tracker.frames.current_frame_idx,
                        name=key,
                        value=value,
                    )
                )
        # Set the current frame to the slots from the latest UserUttered
        events.append(CurrentFrameDumped())
        return events

    def handle_inform(
        self, tracker: "DialogueStateTracker", framed_slots: Dict[Text, Any]
    ) -> List[Event]:
        for key, value in framed_slots.items():
            logger.debug(f"Current frame: {tracker.frames.current_frame}")
            if tracker.frames.current_frame[key] != value:
                logger.debug("Created and switched to new frame")
                return [FrameCreated(slots=framed_slots, switch_to=True)]
        return []

    def handle_switch_frame(
        self, tracker: "DialogueStateTracker", framed_slots: Dict[Text, Any]
    ) -> List[Event]:
        equality_counts = Counter()
        for key, value in framed_slots.items():
            # If the slot value from the latest utterance is not equal to that of the
            # current_frame, search for it among the other frames.
            if tracker.frames.current_frame[key] == value:
                continue
            for idx, frame in enumerate(tracker.frames):
                if idx == tracker.frames.current_frame_idx:
                    continue
                if frame[key] == value:
                    equality_counts[idx] += 1
        # If all the slots mentioned in the latest utterance are matching inside the
        # top-ranking frame, switch to that frame. Otherwise, switch to the most recently
        # created frame.
        best_matches = equality_counts.most_common()
        if best_matches and best_matches[0][1] == len(framed_slots):
            return [CurrentFrameChanged(frame_idx=best_matches[0][0])]
        else:
            most_recent_frames = list(
                sorted(tracker.frames, key=lambda f: f.last_active, reverse=True)
            )
            return [CurrentFrameChanged(frame_idx=most_recent_frames[0].idx)]

    def handle_act_with_ref(
        self, tracker: "DialogueStateTracker", framed_slots: Dict[Text, Any]
    ) -> List[Event]:
        equality_counts = Counter()
        for key, slot in framed_slots.items():
            for idx, frame in enumerate(tracker.frames):
                if frame[key] == slot.value:
                    equality_counts[idx] += 1
        best_matches = equality_counts.most_common()
        if len(best_matches) == 1:
            # If just one match, check if it matches all slots and then set ref to that frame.
            if best_matches[0][1] == len(framed_slots):
                return [
                    FrameUpdated(
                        frame_idx=tracker.frames.current_frame_idx,
                        name="ref",
                        value=best_matches[0][0],
                    )
                ]
        elif len(best_matches) > 1:
            # If more than one best match, set ref to the most recently created of them.
            if best_matches[0][1] == best_matches[1][1] == len(framed_slots):
                most_recent_frame_idx = list(
                    sorted(
                        best_matches,
                        key=lambda x: tracker.frames[x[0]].created,
                        reverse=True,
                    )
                )[0][0]
                return [
                    FrameUpdated(
                        frame_idx=tracker.frames.current_frame_idx,
                        name="ref",
                        value=most_recent_frame_idx,
                    )
                ]
        else:
            # Otherwise, set ref to current frame
            return [
                FrameUpdated(
                    frame_idx=tracker.frames.current_frame_idx,
                    name="ref",
                    value=tracker.frames.current_frame_idx,
                )
            ]
