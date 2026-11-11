"""tools/pr_watch.py: which PRs the review agent must act on (pure logic, no gh calls)."""

import importlib.util

import pytest

from tests.conftest import REPO_ROOT

_spec = importlib.util.spec_from_file_location("pr_watch", REPO_ROOT / "tools" / "pr_watch.py")
pr_watch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr_watch)

ME = "dakshcodez"
OTHER = "prakartii"
PASS = [{"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SUCCESS"}]
FAIL = [{"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "FAILURE"}]


def make_pr(number, author, sha="aaa", *, draft=False, reviews=0, comments=0, checks=None):
    return {
        "number": number,
        "author": {"login": author},
        "headRefOid": sha,
        "title": f"PR {number}",
        "isDraft": draft,
        "reviews": [{}] * reviews,
        "comments": [{}] * comments,
        "statusCheckRollup": PASS if checks is None else checks,
    }


def keys(items):
    return [i["key"] for i in items]


def test_pending_items_new_pr_from_other_person_needs_review():
    items = pr_watch.pending_items([make_pr(7, OTHER, "abc")], ME, set())

    assert keys(items) == ["review:7:abc"]
    assert items[0]["kind"] == "review"


def test_pending_items_reviewed_head_commit_is_not_repeated():
    items = pr_watch.pending_items([make_pr(7, OTHER, "abc")], ME, {"review:7:abc"})

    assert items == []


def test_pending_items_new_commit_after_review_needs_review_again():
    items = pr_watch.pending_items([make_pr(7, OTHER, "def")], ME, {"review:7:abc"})

    assert keys(items) == ["review:7:def"]


def test_pending_items_draft_from_other_person_is_skipped():
    assert pr_watch.pending_items([make_pr(7, OTHER, draft=True)], ME, set()) == []


def test_pending_items_own_pr_without_feedback_is_skipped():
    assert pr_watch.pending_items([make_pr(8, ME)], ME, set()) == []


def test_pending_items_own_pr_with_new_comment_is_reported():
    # 1 review + 2 comments = 3 feedback items on head "s1", checks passing.
    items = pr_watch.pending_items([make_pr(8, ME, "s1", reviews=1, comments=2)], ME, set())

    assert keys(items) == ["own:8:s1:3:passing"]


def test_pending_items_own_pr_reported_again_when_feedback_grows():
    seen = {"own:8:s1:3:passing"}
    items = pr_watch.pending_items([make_pr(8, ME, "s1", reviews=1, comments=3)], ME, seen)

    assert keys(items) == ["own:8:s1:4:passing"]


def test_pending_items_own_pr_with_failing_checks_is_reported():
    items = pr_watch.pending_items([make_pr(8, ME, "s1", checks=FAIL)], ME, set())

    assert keys(items) == ["own:8:s1:0:failing"]


def test_pending_items_are_ordered_by_pr_number():
    prs = [make_pr(9, OTHER, "b"), make_pr(3, OTHER, "a")]

    assert keys(pr_watch.pending_items(prs, ME, set())) == ["review:3:a", "review:9:b"]


@pytest.mark.parametrize(
    ("rollup", "expected"),
    [
        ([], "none"),
        (PASS, "passing"),
        (FAIL, "failing"),
        ([{"__typename": "CheckRun", "status": "IN_PROGRESS", "conclusion": None}], "pending"),
        ([{"__typename": "StatusContext", "state": "ERROR"}], "failing"),
        ([{"__typename": "StatusContext", "state": "PENDING"}], "pending"),
        # One failure wins over passes and pending runs.
        ([*PASS, {"__typename": "CheckRun", "status": "QUEUED"}, *FAIL], "failing"),
        (
            [*PASS, {"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SKIPPED"}],
            "passing",
        ),
    ],
)
def test_check_state_summarizes_rollup(rollup, expected):
    assert pr_watch.check_state(rollup) == expected
