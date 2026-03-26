"""Tests for the /api/v1/movies endpoints."""

from __future__ import annotations


def test_list_movies(client):
    """Paginated movie listing."""
    resp = client.get("/api/v1/movies")
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "meta" in body
    assert body["meta"]["page"] == 1
    assert body["meta"]["total_items"] >= 1


def test_list_movies_genre_filter(client):
    """Filter movies by genre."""
    resp = client.get("/api/v1/movies?genre=Comedy")
    assert resp.status_code == 200
    body = resp.json()
    for item in body["items"]:
        assert "Comedy" in item["genres"]


def test_get_movie_detail(client):
    """Fetch a single movie's detail."""
    resp = client.get("/api/v1/movies/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == 1
    assert "title" in body
    assert "num_ratings" in body
    assert "imdb_id" in body


def test_get_movie_not_found(client):
    """Unknown movie returns 404."""
    resp = client.get("/api/v1/movies/999999")
    assert resp.status_code == 404


def test_search_movies(client):
    """Title search returns matching movies."""
    resp = client.get("/api/v1/movies/search?q=Toy")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) >= 1
    assert "Toy" in body[0]["title"]


def test_search_movies_no_results(client):
    """Search with no matches returns empty list."""
    resp = client.get("/api/v1/movies/search?q=xyznonexistent")
    assert resp.status_code == 200
    assert resp.json() == []


def test_search_movies_missing_query(client):
    """Search without q parameter fails validation."""
    resp = client.get("/api/v1/movies/search")
    assert resp.status_code == 422
