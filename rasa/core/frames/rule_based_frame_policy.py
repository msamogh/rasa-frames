import logging
from typing import List, Dict, Text, Any, Callable

from rasa.core.events import Event, FrameCreated, CurrentFrameChanged
from rasa.core.frames import FramePolicy
from rasa.core.frames.frame_policy import FrameIntent


logger = logging.getLogger(__name__)


def most_recent_frame_idx(frames: List["Frame"]) -> int:
    most_recent_frame = list(
        sorted(frames, key=lambda frame: frame.created, reverse=True)
    )[0]
    return most_recent_frame.idx


def choice_fn_by_name(name: Text):
    fns = {
        'create_new': create_new,
        'most_recent': most_recent,
    }
    return fns[name]


def create_new(
    matching_candidates: List["Frame"],
    non_conflicting_candidates: List["Frame"],
    all_frames: List["Frame"]
) -> int:
    """Create a new frame if no perfect match found."""
    if matching_candidates:
        return most_recent_frame_idx(matching_candidates)
    if non_conflicting_candidates:
        if all_frames.current_frame_idx in non_conflicting_candidates:
            return all_frames.current_frame_idx
    return len(all_frames)


def most_recent(
    matching_candidates: List["Frame"],
    non_conflicting_candidates: List["Frame"],
    all_frames: List["Frame"]
) -> int:
    if matching_candidates:
        return most_recent_frame_idx(matching_candidates)
    return most_recent_frame_idx(all_frames)


class RuleBasedFramePolicy(FramePolicy):

    def get_best_matching_frame_idx(
        self,
        frames: List["Frame"],
        frame_intent: FrameIntent,
        framed_entities: Dict[Text, Any]
    ) -> int:
        """Return the most recent frame for which all slots match.

        If no frame inside frames exists such that there's a full match,
        return None. If there are multiple matches, return the most recent
        of the matching frames.
        """
        from collections import Counter

        assert len(frames) > 0

        equality_counts = Counter()
        conflict_counts = Counter()
        for key, value in framed_entities.items():
            for idx, frame in enumerate(frames):
                if frame[key] == value:
                    equality_counts[idx] += 1
                else:
                    conflict_counts[idx] += 1

        frames_sorted_by_num_equal = equality_counts.most_common()
        frames_sorted_by_num_conflicts = conflict_counts.most_common()
        # If the best match has a less-than-perfect match, simply return
        # the most recently created frame.
        matching_candidates = []
        for idx, num_matches in frames_sorted_by_num_equal:
            if num_matches == len(framed_entities):
                matching_candidates.append(frames[idx])
        non_conflicting_candidates = []
        for idx, num_conflicts in frames_sorted_by_num_conflicts:
            if num_conflicts == 0:
                non_conflicting_candidates.append(frames[idx])

        choice_fn = choice_fn_by_name(frame_intent.frame_ref_fallback)
        return choice_fn(
            matching_candidates,
            non_conflicting_candidates,
            frames
        )

    def on_frame_ref_identified(
        self,
        tracker: "DialogueStateTracker",
        ref_frame_idx: int,
        frame_intent: FrameIntent
    ) -> List[Event]:
        assert ref_frame_idx <= len(tracker.frames), \
            "ref cannot violate 0 <= ref <= len(tracker.frames)"

        if frame_intent.on_frame_ref_identified == 'switch':
            if ref_frame_idx == len(tracker.frames):
                return [FrameCreated(
                    slots=framed_entities,
                    switch_to=True
                )]
            else:
                return [CurrentFrameChanged(
                    frame_idx=ref_frame_idx
                )]
        elif on_frame_ref_identified == 'nothing':
            pass
