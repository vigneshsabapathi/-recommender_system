"""Tests for the /api/v1/similar endpoint."""

from __future__ import annotations


def test_get_similar_default(client):
    """Default similar-movies response."""
    resp = client.get("/api/v1/similar/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["movie_id"] == 1
    assert body["algorithm"] == "collaborative"
    assert len(body["similar"]) >= 1
    assert body["metadata"]["total"] >= 1


def test_get_similar_with_algorithm(client):
    """Specify an explicit algorithm."""
    resp = client.get("/api/v1/similar/1?algorithm=content_based&n=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["algorithm"] == "content_based"
    assert len(body["similar"]) == 1


def test_get_similar_movie_fields(client):
    """Each similar movie has card metadata."""
    resp = client.get("/api/v1/similar/1")
    assert resp.status_code == 200
    entry = resp.json()["similar"][0]
    assert "movie" in entry
    assert "score" in entry
    assert entry["movie"]["id"] > 0


def test_get_similar_invalid_n(client):
    """n=0 is rejected."""
    resp = client.get("/api/v1/similar/1?n=0")
    assert resp.status_code == 422
