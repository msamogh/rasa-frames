import logging
from collections import Counter
from typing import List, Text, Tuple, Dict, Optional

from rasa.core.trackers import DialogueStateTracker
from rasa.core.slots import Slot


logger = logging.getLogger(__name__)


class FrameSet(object):

    def __init__(self, init_slots: Dict[Slot]) -> None:
        self.frames = []

        init_slots = FrameSet.get_framed_slots(init_slots)
        self.frames.append(Frame(0, init_slots))

    def add_frame(
        self, slots: List[Slot], created: Optional[int], last_active: Optional[int]
    ) -> Frame:
        frame = Frame(len(self.frames), FrameSet.get_framed_slots(slots))
        self.frames.append(frame)
        return frame

    @staticmethod
    def get_framed_slots(slots: Dict[Slot]) -> Dict[Slot]:
        return {
            name: slot for name, slot in slots.items()
            if slot.frame_slot and slot.value is not None
        }


class Frame(object):

    def __init__(self, idx: int, slots: Dict[Slots]) -> None:
        self.slots = slots
        self.idx = idx
        self.created = None
        self.last_active = None

    def __getitem__(self, key: Text) -> Optional[Any]:
        return self.slots.get(key, None)

    def __setitem__(self, key: Text, value: Any) -> None:
        self.slots[key] = value


class RuleBasedFrameTracker(object):

    def update_frames(
        self, tracker: DialogueStateTracker, user_utterance: Text
    ) -> None:
        # Treat the slot values in the tracker as temporary values
        # (not necessarily reflecting the values of the active frame).
        # The active frame will be decided upon only after checking with the FrameTracker.
        dialogue_act = user_utterance.intent
        frames = tracker.frames
        current_frame = tracker.current_frame

        if dialogue_act == 'inform':
            self.handle_inform(tracker)
        elif dialogue_act == 'switch_frame':
            self.handle_switch_frame(tracker)
        elif dialogue_act in ['']:  # anything with a ref tag
            self.handle_act_with_ref(tracker, dialogue_act)

    def handle_switch_frame(self, tracker: DialogueStateTracker) -> None:
        equality_counts = Counter()
        framed_slots = list(FrameSet.get_framed_slots(tracker.slots).items())
        for key, slot in framed_slots:
            # If the slot value from the latest utterance is not equal to that of the
            # current_frame, search for it among the other frames.
            if frames[current_frame][key] == slot.value:
                continue
            for idx, frame in enumerate(frames):
                if idx == current_frame:
                    continue
                if frame[key] == slot.value:
                    equality_counts[idx] += 1
        # If all the slots mentioned in the latest utterance are matching inside the
        # top-ranking frame, switch to that frame. Otherwise, switch to the most recently
        # created frame.
        best_match = equality_counts.most_common()
        if best_match and best_match[0][1] == len(framed_slots):
            tracker.current_frame = best_match[0][0]
            tracker.frames[current_frame].last_active =  # TODO
        else:
            most_recent_frames = list(
                sorted(tracker.frames, key=lambda f: f.last_active, reverse=True))
            tracker.current_frame = most_recent_frames[0].idx

    def handle_inform(self, tracker: DialogueStateTracker) -> None:
        framed_slots = FrameSet.get_framed_slots(tracker.slots)
        for key, slot in framed_slots.items():
            if tracker.frames[tracker.current_frame][key] != tracker.slots[key].value:
                break
        else:
            frame = tracker.frames.add_frame(framed_slots)
            tracker.frames[current_frame] = frame.idx

    def handle_act_with_ref(
        self, tracker: DialogueStateTracker, dialogue_act: Text
    ) -> None:
        pass
