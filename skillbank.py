"""SkillBank utilities for GeoSkill-RL."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .spatial import parse_locator


SEED_SKILLS: list[dict[str, Any]] = [
    {
        "skill_id": "top_extremum",
        "trigger": ["top-most", "uppermost", "upper", "north", "northernmost"],
        "coordinate_prior": "smaller y-center",
        "principle": "When several same-class candidates are plausible, rank them by smaller y-center before size or salience.",
        "when_to_apply": "Use for questions asking for the upper or northern instance of an object.",
        "avoid": "Do not select a lower but larger or more visually salient instance.",
        "covered_failure_types": ["top_violated"],
    },
    {
        "skill_id": "bottom_extremum",
        "trigger": ["bottom-most", "lowermost", "lower", "south", "southernmost"],
        "coordinate_prior": "larger y-center",
        "principle": "When several same-class candidates are plausible, rank them by larger y-center before size or salience.",
        "when_to_apply": "Use for questions asking for the lower or southern instance of an object.",
        "avoid": "Do not select an upper but larger or more central instance.",
        "covered_failure_types": ["bottom_violated"],
    },
    {
        "skill_id": "left_extremum",
        "trigger": ["left-most", "leftmost", "left", "west", "westernmost"],
        "coordinate_prior": "smaller x-center",
        "principle": "When several same-class candidates are plausible, rank them by smaller x-center before size or salience.",
        "when_to_apply": "Use for questions asking for the left or western instance of an object.",
        "avoid": "Do not select a more central instance if a plausible leftmost candidate exists.",
        "covered_failure_types": ["left_violated"],
    },
    {
        "skill_id": "right_extremum",
        "trigger": ["right-most", "rightmost", "right", "east", "easternmost"],
        "coordinate_prior": "larger x-center",
        "principle": "When several same-class candidates are plausible, rank them by larger x-center before size or salience.",
        "when_to_apply": "Use for questions asking for the right or eastern instance of an object.",
        "avoid": "Do not select a more central instance if a plausible rightmost candidate exists.",
        "covered_failure_types": ["right_violated"],
    },
    {
        "skill_id": "corner_locator",
        "trigger": ["upper-left", "upper right", "lower-left", "lower right", "northwest", "northeast", "southwest", "southeast"],
        "coordinate_prior": "satisfy both x and y directions",
        "principle": "For corner locators, satisfy the vertical and horizontal coordinate priors together before judging salience.",
        "when_to_apply": "Use when a question combines vertical and horizontal locator words.",
        "avoid": "Do not stop after satisfying only one axis of a corner instruction.",
        "covered_failure_types": ["corner_partial", "top_violated", "bottom_violated", "left_violated", "right_violated"],
    },
]


def load_skillbank(path: str | None = None) -> list[dict[str, Any]]:
    if not path:
        return list(SEED_SKILLS)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SkillBank not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("skills", [])
    if not isinstance(data, list):
        raise ValueError(f"SkillBank must be a list or dict with 'skills': {path}")
    return data


def save_skillbank(path: str, skills: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 0, "skills": skills}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _trigger_hit(question: str, trigger: str) -> bool:
    t = re.escape(trigger.lower()).replace("\\-", "[- ]?")
    return re.search(rf"(?<![a-z0-9]){t}(?![a-z0-9])", question.lower()) is not None


def retrieve_skills(question: str, skillbank: list[dict[str, Any]], max_skills: int = 2) -> list[dict[str, Any]]:
    locator = parse_locator(question)
    wanted = set(locator.get("axes") or [])
    if locator.get("family") == "corner":
        wanted.add("corner")

    scored: list[tuple[int, dict[str, Any]]] = []
    q = question or ""
    for skill in skillbank:
        triggers = [str(x) for x in skill.get("trigger", [])]
        hit = any(_trigger_hit(q, t) for t in triggers)
        covered = " ".join(str(x) for x in skill.get("covered_failure_types", []))
        axis_hit = any(axis in covered or axis in str(skill.get("skill_id", "")) for axis in wanted)
        if hit or axis_hit:
            score = (2 if hit else 0) + (1 if axis_hit else 0)
            scored.append((score, skill))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("skill_id", ""))))
    return [s for _, s in scored[:max_skills]]


def format_skill_block(skills: list[dict[str, Any]]) -> str:
    if not skills:
        return ""
    lines = ["Relevant spatial grounding skills:"]
    for skill in skills:
        lines.append(
            "- "
            + str(skill.get("skill_id", "skill"))
            + ": prior="
            + str(skill.get("coordinate_prior", ""))
            + "; apply="
            + str(skill.get("when_to_apply", ""))
            + "; avoid="
            + str(skill.get("avoid", ""))
        )
    lines.append("Use these only to choose the first <zoom> bbox; keep the required XML-like protocol.")
    return "\n".join(lines)
