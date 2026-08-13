"""The identity stack: staying locked on one person, including from behind.

The scenario driving all of this: face recognition confirms the operator, then
they turn around and walk away. From that point the robot only ever sees their
back, so the face layer is useless and the appearance layer has to carry it.
"""

from __future__ import annotations

import numpy as np
import pytest

from spot_voice.vision import appearance as appearance_module
from spot_voice.vision.appearance import AppearanceSignature, blend, describe, similarity
from spot_voice.vision.faces import FaceStore, cosine_similarity, face_inside
from spot_voice.vision.identity import IdentityTracker, LockState, iou, most_prominent

FRAME_W, FRAME_H = 640, 480


def make_frame(*people) -> np.ndarray:
    """Paint a frame with coloured rectangles standing in for people.

    Each entry is ``(box, torso_bgr, legs_bgr)``.
    """
    frame = np.full((FRAME_H, FRAME_W, 3), 30, dtype=np.uint8)
    for box, torso, legs in people:
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        split = y1 + int((y2 - y1) * 0.55)
        frame[y1:split, x1:x2] = torso
        frame[split:y2, x1:x2] = legs
    return frame


def person(centre_x: float, height_fraction: float, confidence: float = 0.9):
    height = height_fraction * FRAME_H
    width = height * 0.4
    return (
        int(centre_x - width / 2),
        int(FRAME_H / 2 - height / 2),
        int(centre_x + width / 2),
        int(FRAME_H / 2 + height / 2),
        confidence,
    )


BLUE_SHIRT = (200, 60, 40)
DARK_TROUSERS = (50, 45, 40)
RED_SHIRT = (40, 50, 200)
GREEN_SHIRT = (60, 180, 70)


# ----------------------------------------------------------------------
# Appearance: the layer that works from behind


def test_the_same_clothing_matches_itself():
    box = person(320, 0.5)
    frame = make_frame((box, BLUE_SHIRT, DARK_TROUSERS))
    signature = describe(frame, box[:4])

    assert signature.valid
    assert similarity(signature, signature) == pytest.approx(1.0, abs=1e-6)


def test_different_clothing_does_not_match():
    box = person(320, 0.5)
    mine = describe(make_frame((box, BLUE_SHIRT, DARK_TROUSERS)), box[:4])
    theirs = describe(make_frame((box, RED_SHIRT, DARK_TROUSERS)), box[:4])

    assert similarity(mine, theirs) < appearance_module.MATCH_THRESHOLD


def test_a_signature_survives_the_person_turning_around():
    # A back view is the same clothing seen from the other side, so the colour
    # signature holds where a face recogniser has nothing to work with.
    box = person(320, 0.5)
    facing = describe(make_frame((box, BLUE_SHIRT, DARK_TROUSERS)), box[:4])
    # Same clothes, slightly different framing and shade, as when walking away.
    away_box = person(315, 0.48)
    away = describe(make_frame((away_box, (190, 55, 38), (48, 43, 38))), away_box[:4])

    assert similarity(facing, away) >= appearance_module.MATCH_THRESHOLD


def test_a_degenerate_box_yields_no_signature():
    frame = make_frame()
    assert not describe(frame, (10, 10, 11, 11)).valid
    assert not describe(None, (0, 0, 50, 100)).valid


def test_blending_tracks_light_change_without_jumping_identity():
    box = person(320, 0.5)
    stored = describe(make_frame((box, BLUE_SHIRT, DARK_TROUSERS)), box[:4])
    dimmer = describe(make_frame((box, (170, 50, 34), (42, 38, 34))), box[:4])
    other = describe(make_frame((box, GREEN_SHIRT, DARK_TROUSERS)), box[:4])

    updated = blend(stored, dimmer)

    assert similarity(updated, stored) > similarity(updated, other)


def test_blending_handles_empty_signatures():
    real = describe(make_frame((person(320, 0.5), BLUE_SHIRT, DARK_TROUSERS)), person(320, 0.5)[:4])
    assert blend(AppearanceSignature(), real) is real
    assert blend(real, AppearanceSignature()) is real


def test_greyscale_frames_still_produce_a_signature():
    # Spot's body fisheyes are greyscale, so this path is the real one on the
    # robot -- weaker than colour, but it must not return nothing.
    box = person(320, 0.5)
    grey = np.full((FRAME_H, FRAME_W), 30, dtype=np.uint8)
    grey[box[1]:box[3], box[0]:box[2]] = 200

    assert describe(grey, box[:4]).valid


# ----------------------------------------------------------------------
# Face store


def test_enrollment_round_trips_through_disk(tmp_path):
    store = FaceStore(tmp_path / "faces.json")
    store.add("umar", [1.0, 0.0, 0.0])
    store.add("umar", [0.9, 0.1, 0.0])
    store.save()

    reloaded = FaceStore(tmp_path / "faces.json")
    assert reloaded.names == ["umar"]
    name, score = reloaded.identify([1.0, 0.0, 0.0])
    assert name == "umar" and score == pytest.approx(1.0)


def test_an_unknown_face_is_not_identified(tmp_path):
    store = FaceStore(tmp_path / "faces.json")
    store.add("umar", [1.0, 0.0, 0.0])

    name, score = store.identify([0.0, 1.0, 0.0])
    assert name is None
    assert score < 0.42


def test_a_corrupt_store_starts_empty_rather_than_crashing(tmp_path):
    path = tmp_path / "faces.json"
    path.write_text("{not json", encoding="utf-8")
    assert FaceStore(path).is_empty


def test_forgetting_someone(tmp_path):
    store = FaceStore(tmp_path / "faces.json")
    store.add("umar", [1.0, 0.0])
    assert store.forget("umar") is True
    assert store.forget("nobody") is False


def test_cosine_similarity_edge_cases():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_a_face_is_attributed_to_the_body_it_sits_in():
    body = person(320, 0.6)
    face_box = (310, body[1] + 10, 330, body[1] + 40)
    assert face_inside(face_box, body) is True
    assert face_inside((10, 10, 30, 40), body) is False


# ----------------------------------------------------------------------
# Geometry helpers


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_most_prominent_prefers_near_and_centred():
    near_centre = person(320, 0.6)
    far_edge = person(40, 0.2)
    assert most_prominent([far_edge, near_centre], FRAME_W) == near_centre


# ----------------------------------------------------------------------
# The tracker: the whole point


def test_the_walking_away_scenario():
    """Face confirms, operator turns and walks off, tracking continues."""
    tracker = IdentityTracker(operator="umar")

    # 1. Facing the robot. Face recognised, lock established.
    box = person(320, 0.55)
    frame = make_frame((box, BLUE_SHIRT, DARK_TROUSERS))
    tracker.acquire(frame, box, name="umar", by_face=True)
    assert tracker.locked and tracker.confirmed_by_face
    assert tracker.target_name == "umar"

    # 2. Turned around, walking away. No face available at all -- faces=None.
    for step, (centre, height) in enumerate(
        [(322, 0.53), (326, 0.50), (330, 0.47), (334, 0.44)]
    ):
        moved = person(centre, height)
        frame = make_frame((moved, BLUE_SHIRT, DARK_TROUSERS))
        seen = tracker.update(frame, [moved], faces=None, face_store=None)
        assert seen == moved, f"lost the operator at step {step}"
        assert tracker.state is LockState.TRACKING


def test_a_crosser_in_different_clothes_cannot_steal_the_follow():
    tracker = IdentityTracker()
    mine = person(320, 0.5)
    frame = make_frame((mine, BLUE_SHIRT, DARK_TROUSERS))
    tracker.acquire(frame, mine, by_face=True)

    # Someone in a red shirt steps between us: closer to the camera, so bigger,
    # and overlapping. Only the appearance layer can tell them apart.
    crosser = person(325, 0.95)
    me_now = person(324, 0.5)
    frame = make_frame((me_now, BLUE_SHIRT, DARK_TROUSERS), (crosser, RED_SHIRT, RED_SHIRT))

    assert tracker.update(frame, [crosser, me_now]) == me_now


def test_full_occlusion_holds_rather_than_retargeting():
    tracker = IdentityTracker()
    mine = person(320, 0.5)
    tracker.acquire(make_frame((mine, BLUE_SHIRT, DARK_TROUSERS)), mine, by_face=True)

    # Completely hidden: only the crosser is detected.
    crosser = person(322, 0.95)
    frame = make_frame((crosser, RED_SHIRT, RED_SHIRT))

    assert tracker.update(frame, [crosser]) is None
    assert tracker.state is LockState.OCCLUDED
    assert tracker.locked  # still ours; we are waiting, not re-targeting


def test_appearance_refinds_after_geometry_breaks():
    """Round a corner, reappear somewhere else: geometry fails, colour saves it."""
    tracker = IdentityTracker()
    mine = person(320, 0.5)
    tracker.acquire(make_frame((mine, BLUE_SHIRT, DARK_TROUSERS)), mine, by_face=True)

    # Reappears far away with no overlap at all -- geometry cannot bridge that.
    elsewhere = person(120, 0.5)
    frame = make_frame((elsewhere, BLUE_SHIRT, DARK_TROUSERS))

    assert iou(elsewhere, mine) == 0.0
    assert tracker.update(frame, [elsewhere]) == elsewhere


def test_appearance_does_not_refind_a_different_person():
    tracker = IdentityTracker()
    mine = person(320, 0.5)
    tracker.acquire(make_frame((mine, BLUE_SHIRT, DARK_TROUSERS)), mine, by_face=True)

    stranger = person(120, 0.5)
    frame = make_frame((stranger, GREEN_SHIRT, GREEN_SHIRT))

    assert tracker.update(frame, [stranger]) is None


def test_the_lock_drops_after_the_lost_timeout():
    from spot_voice.vision import identity as identity_module

    tracker = IdentityTracker()
    mine = person(320, 0.5)
    tracker.acquire(make_frame((mine, BLUE_SHIRT, DARK_TROUSERS)), mine, by_face=True)

    # Backdate the sighting past the timeout instead of sleeping.
    tracker._target.last_seen -= identity_module.LOST_AFTER_SEC + 0.1

    assert tracker.update(make_frame(), []) is None
    assert not tracker.locked
    assert tracker.state is LockState.SEARCHING


def test_a_recognised_face_reconfirms_and_wins_over_geometry():
    class Store:
        def identify(self, _embedding, threshold=0.42):
            return "umar", 0.9

    tracker = IdentityTracker(operator="umar")
    mine = person(320, 0.5)
    tracker.acquire(make_frame((mine, BLUE_SHIRT, DARK_TROUSERS)), mine, by_face=False)

    # Operator turns round: a face is now visible on a box geometry would reject.
    turned = person(500, 0.5)
    frame = make_frame((turned, BLUE_SHIRT, DARK_TROUSERS))
    face_box = (495, turned[1] + 10, 515, turned[1] + 40)

    seen = tracker.update(frame, [turned], faces=[(face_box, [1.0])], face_store=Store())

    assert seen == turned
    assert tracker.confirmed_by_face is True
    assert tracker.target_name == "umar"


def test_a_face_belonging_to_someone_else_is_ignored():
    class Store:
        def identify(self, _embedding, threshold=0.42):
            return "awaiz", 0.9

    tracker = IdentityTracker(operator="umar")
    other = person(500, 0.5)
    frame = make_frame((other, RED_SHIRT, RED_SHIRT))
    face_box = (495, other[1] + 10, 515, other[1] + 40)

    seen = tracker.update(frame, [other], faces=[(face_box, [1.0])], face_store=Store())
    assert seen is None
    assert not tracker.locked


def test_fallback_acquisition_when_no_face_is_visible():
    tracker = IdentityTracker()
    near = person(320, 0.6)
    far = person(60, 0.2)
    frame = make_frame((near, BLUE_SHIRT, DARK_TROUSERS), (far, GREEN_SHIRT, GREEN_SHIRT))

    assert tracker.acquire_fallback(frame, [far, near], FRAME_W) == near
    assert tracker.locked
    assert tracker.confirmed_by_face is False


def test_release_returns_to_searching():
    tracker = IdentityTracker()
    mine = person(320, 0.5)
    tracker.acquire(make_frame((mine, BLUE_SHIRT, DARK_TROUSERS)), mine)
    tracker.release()
    assert not tracker.locked
    assert tracker.state is LockState.SEARCHING
