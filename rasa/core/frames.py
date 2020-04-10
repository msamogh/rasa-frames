import logging
from collections import Counter
from typing import Any, List, Text, Tuple, Dict, Optional

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
        self, tracker: "DialogueStateTracker", user_utterance: Text
    ) -> List[Event]:
        dialogue_entities = FrameSet.get_framed_entities(
            user_utterance.entities, self.domain
        )

        # 1. Update the current frame with slots that might have been updated
        # through other custom actions.
        # 2. Check for changes in either the slot-values within the current frame
        # or for a switching of the active frame.
        # 3. Refill the slots with the contents of the current frame
        events = push_slots_into_current_frame(tracker)
                 + self._handle_frame_changes(tracker, dialogue_entities, events)
                 + pop_slots_from_current_frame()

        return events

    def _handle_frame_changes(
        self,
        tracker: "DialogueStateTracker",
        dialogue_entities: Dict[Text, Any],
        events: List[Event]
    ) -> List[Event]:
        # Populate the ref field in dialogue_entities
        self._populate_utterance_with_ref(tracker, dialogue_entities, events)

        intent_class = self.domain.intent_properties.get(
            user_utterance.intent["name"]
        ).get("intent_class")

        assert intent_class in ["constraint", "request", "comparison_request", "binary_question", None]
        if intent_class == "constraint":
            logger.debug("Handling inform")
            events.extend(self._handle_inform(tracker, dialogue_entities, events))
        elif intent_class == "switch_frame":
            logger.debug("Switching frame")
            events.extend(self._handle_switch_frame(tracker, dialogue_entities, events))
        elif intent_class in acts_with_ref:
            # anything with a ref tag
            logger.debug("Handling act with ref")
            events.extend(self._handle_act_with_ref(tracker, dialogue_entities, events))
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
        return events

    def _populate_utterance_with_ref(
        self,
        tracker: "DialogueStateTracker",
        framed_entities: Dict[Text, Any],
        events_this_turn: List[Event]
    ) -> None:
        first_frame_created_now = is_first_frame_created_now(
            tracker, events_this_turn
        )

        # Get all the candidate frames.
        # If the first frame was created recently, then fetch it from
        # events_this_turn. Otherwise, fetch it from tracker.frames.
        if not tracker.frames.current_frame:
            assert first_frame_created_now, "Either the current frame must not "
                        "be None or a frame must be created recently."
            # Set all_frames to the first frame that was just created earlier
            all_frames = [Frame(
                slots=events_this_turn[0].slots,
                idx=0,
                created=events_this_turn[0].timestamp
            )]
        else:
            all_frames = tracker.frames

        # Fetches the frame id of the most recent frame having the most number of
        # key-value matches against the entities picked up during this turn.
        # If no best match is found, then the ref is set to the current frame's id.
        current_frame_idx = 0 if first_frame_created_now \
                                else tracker.frames.current_frame_idx
        framed_entities['ref'] = get_best_matching_frame_idx(
            all_frames, framed_entities, fallback_idx=current_frame_idx
        )



    def _handle_inform(
        self,
        tracker: "DialogueStateTracker",
        framed_entities: Dict[Text, Any],
        events_this_turn: List[Event],
    ) -> List[Event]:
        logger.debug(f"Received the great framed_entities: {framed_entities}")
        for key, value in framed_entities.items():
            logger.debug(f"Current frame: {tracker.frames.current_frame}")
            if tracker.frames.current_frame:
                #
                if tracker.frames.current_frame[key] is None:
                    logger.debug("Updating current frame")
                    return [
                        FrameUpdated(
                            frame_idx=tracker.frames.current_frame_idx,
                            name=key,
                            value=value,
                        )
                        for key, value in framed_entities.items()
                    ]
                elif tracker.frames.current_frame[key] != value:
                    logger.debug("Created and switched to new frame")
                    return [FrameCreated(slots=framed_entities, switch_to=True)]
            elif is_first_frame_created_now(tracker, events_this_turn):
                logger.debug(
                    "First frame has already been created. So just updating it now..."
                )
                return [
                    FrameUpdated(frame_idx=0, name=key, value=value)
                    for key, value in framed_entities.items()
                ]
            else:
                logger.debug("Created and switched to new frame 2")
                return [FrameCreated(slots=framed_entities, switch_to=True)]
        return []

    def _handle_switch_frame(
        self,
        tracker: "DialogueStateTracker",
        framed_entities: Dict[Text, Any],
        events_this_turn: List[Event],
    ) -> List[Event]:
        equality_counts = Counter()
        for key, value in framed_entities.items():
            # If the slot value from the latest utterance is not equal to that of the
            # current_frame, search for it among the other frames.
            if (
                tracker.frames.current_frame
                and tracker.frames.current_frame[key] == value
            ):
                # The only reason tracker.frames.current_frame is being checked here
                # is to ensure non-nullity.
                continue
            for idx, frame in enumerate(tracker.frames.frames):
                if idx == tracker.frames.current_frame_idx:
                    continue
                if frame[key] == value:
                    equality_counts[idx] += 1
        # If all the slots mentioned in the latest utterance are matching inside the
        # top-ranking frame, switch to that frame. Otherwise, switch to the most recently
        # created frame.
        best_matches = equality_counts.most_common()
        if best_matches and best_matches[0][1] == len(framed_entities):
            logger.debug(f"Found a complete match. Switching to {best_matches[0][0]}")
            return [CurrentFrameChanged(frame_idx=best_matches[0][0])]
        else:
            most_recent_frames = list(
                sorted(tracker.frames, key=lambda f: f.last_active, reverse=True)
            )
            logger.debug(
                f"Could not find a complete match. Switching to most recent"
                f"frame: {most_recent_frames[0].idx}"
            )
            return [CurrentFrameChanged(frame_idx=most_recent_frames[0].idx)]
