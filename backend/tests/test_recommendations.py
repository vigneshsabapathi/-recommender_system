"""Tests for the /api/v1/recommendations endpoint."""

from __future__ import annotations


def test_get_recommendations_default(client):
    """Default hybrid recommendations for a known user."""
    resp = client.get("/api/v1/recommendations/42")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == 42
    assert body["algorithm"] == "hybrid"
    assert len(body["recommendations"]) >= 1
    assert body["metadata"]["total"] >= 1
    assert body["metadata"]["processing_time_ms"] >= 0


def test_get_recommendations_with_algorithm(client):
    """Specify an explicit algorithm."""
    resp = client.get("/api/v1/recommendations/42?algorithm=collaborative&n=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["algorithm"] == "collaborative"
    assert len(body["recommendations"]) == 1


def test_get_recommendations_with_explanation(client):
    """Request explanations."""
    resp = client.get("/api/v1/recommendations/42?explain=true&n=2")
    assert resp.status_code == 200
    body = resp.json()
    rec = body["recommendations"][0]
    assert "explanation" in rec
    # The first recommendation should have an explanation item
    assert len(rec["explanation"]) >= 1
    assert "reason" in rec["explanation"][0]


def test_get_recommendations_movie_fields(client):
    """Verify movie card fields are populated."""
    resp = client.get("/api/v1/recommendations/42")
    assert resp.status_code == 200
    rec = resp.json()["recommendations"][0]
    movie = rec["movie"]
    assert "id" in movie
    assert "title" in movie
    assert isinstance(movie["genres"], list)
    assert rec["score"] > 0


def test_get_recommendations_invalid_n(client):
    """n=0 is rejected by validation."""
    resp = client.get("/api/v1/recommendations/42?n=0")
    assert resp.status_code == 422


def test_get_recommendations_invalid_algorithm(client):
    """Unknown algorithm is rejected."""
    resp = client.get("/api/v1/recommendations/42?algorithm=bogus")
    assert resp.status_code == 422
