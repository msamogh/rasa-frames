import logging
from collections import Counter
from typing import Any, List, Text, Tuple, Dict, Optional

from rasa.core.events import Event, SlotSet, UserUttered
from rasa.core.slots import Slot


logger = logging.getLogger(__name__)


class Frame(object):

    def __init__(
        self,
        idx: int,
        slots: Dict[Text, Slot],
        created: float,
        switch_to: Optional[bool] = False
    ) -> None:
        self.slots = slots
        self.idx = idx
        self.created = created
        self.last_active = created if switch_to else None

    def __getitem__(self, key: Text) -> Optional[Any]:
        return self.slots.get(key, None)

    def __setitem__(self, key: Text, value: Any) -> None:
        self.slots[key] = value


class FrameSet(object):

    def __init__(self, init_slots: Dict[Text, Slot], created: float) -> None:
        self.frames = []

        init_slots = FrameSet.get_framed_slots(init_slots)
        init_frame = Frame(
            idx=0,
            slots=init_slots,
            created=created,
            switch_to=True
        )
        self.frames.append(init_frame)

    def add_frame(
        self, slots: List[Slot], created: float, switch_to: Optional[bool] = False
    ) -> Frame:
        logger.debug(f'Frame created with values {slots}')
        frame = Frame(
            idx=len(self.frames),
            slots=FrameSet.get_framed_slots(slots),
            created=created,
            switch_to=switch_to
        )
        self.frames.append(frame)
        return frame

    def activate_frame(self, idx: int, timestamp: float) -> None:
        self.frames[idx].last_active = timestamp

    @staticmethod
    def get_framed_slots(slots: Dict[Text, Slot]) -> Dict[Text, Slot]:
        return {
            name: slot for name, slot in slots.items()
            if slot.frame_slot and slot.value is not None
        }


class RuleBasedFrameTracker(object):

    def update_frames(
        self, tracker: "DialogueStateTracker", user_utterance: Text
    ) -> List[Event]:
        # Treat the slot values in the tracker as temporary values
        # (not necessarily reflecting the values of the active frame).
        # The active frame will be decided upon only after checking with the FrameTracker.
        dialogue_act = user_utterance.intent
        frames = tracker.frames
        current_frame = tracker.current_frame

        acts_with_ref = ['affirm', 'canthelp', 'confirm', 'hearmore', 'inform',
                         'moreinfo', 'negate', 'no_result', 'offer', 'request', 'request_compare',
                         'suggest', 'switch_frame']
        if dialogue_act == 'inform':
            self.handle_inform(tracker, user_utterance)
        elif dialogue_act == 'switch_frame':
            self.handle_switch_frame(tracker, user_utterance)
        elif dialogue_act in acts_with_ref and tracker.slots['ref'] is not None:
            # anything with a ref tag
            self.handle_act_with_ref(tracker, user_utterance)
        else:
            # Set the current frame to the slots from the latest UserUttered
            for key, slot in FrameSet.get_framed_slots(tracker.slots).items():
                tracker.frames[current_frame][key] = slot

        for key, value in frames[current_frame].items():
            tracker.update(SlotSet(key, value))

    def handle_switch_frame(
        self, tracker: "DialogueStateTracker", user_utterance: UserUttered
    ) -> None:
        equality_counts = Counter()
        framed_slots = list(FrameSet.get_framed_slots(tracker.slots).items())
        for key, slot in framed_slots:
            # If the slot value from the latest utterance is not equal to that of the
            # current_frame, search for it among the other frames.
            if frames[current_frame][key].value == slot.value:
                continue
            for idx, frame in enumerate(frames):
                if idx == current_frame:
                    continue
                if frame[key] == slot.value:
                    equality_counts[idx] += 1
        # If all the slots mentioned in the latest utterance are matching inside the
        # top-ranking frame, switch to that frame. Otherwise, switch to the most recently
        # created frame.
        best_matches = equality_counts.most_common()
        if best_matches and best_matches[0][1] == len(framed_slots):
            tracker.current_frame = best_matches[0][0]
            tracker.frames.activate_frame(
                idx=tracker.current_frame,
                timestamp=user_utterance.timestamp
            )
        else:
            most_recent_frames = list(
                sorted(tracker.frames, key=lambda f: f.last_active, reverse=True))
            tracker.current_frame = most_recent_frames[0].idx
            tracker.frames.activate_frame(
                idx=tracker.current_frame,
                timestamp=user_utterance.timestamp
            )

    def handle_inform(
        self, tracker: "DialogueStateTracker", user_utterance: UserUttered
    ) -> None:
        framed_slots = FrameSet.get_framed_slots(tracker.slots)
        for key, slot in framed_slots.items():
            if tracker.frames[tracker.current_frame][key].value != tracker.slots[key].value:
                break
        else:
            frame = tracker.frames.add_frame(
                framed_slots,
                created=user_utterance.timestamp,
                switch_to=True
            )
            tracker.current_frame = frame.idx

    def handle_act_with_ref(
        self, tracker: "DialogueStateTracker", user_utterance: UserUttered
    ) -> None:
        equality_counts = Counter()
        framed_slots = FrameSet.get_framed_slots(tracker.slots)
        for key, slot in framed_slots.items():
            for idx, frame in enumerate(tracker.frames):
                if frame[key].value == slot.value:
                    equality_counts[idx] += 1
        best_matches = equality_counts.most_common()
        if len(best_matches) == 1:
            # If just one match, check if it matches all slots and then set ref to that frame.
            if best_matches[0][1] == len(framed_slots):
                tracker.frames[tracker.current_frame]['ref'] = best_matches[0][0]
        elif len(best_matches) > 1:
            # If more than one best match, set ref to the most recently created of them.
            if best_matches[0][1] == best_matches[1][1] == len(framed_slots):
                most_recent_frame_idx = list(sorted(best_matches,
                                                    key=lambda x: tracker.slots[x[0]
                                                                                ].created,
                                                    reverse=True))[0][0]
                tracker.frames[tracker.current_frame]['ref'] = most_recent_frame_idx
        else:
            # Otherwise, set ref to current frame
            tracker.frames[tracker.current_frame]['ref'] = tracker.current_frame
