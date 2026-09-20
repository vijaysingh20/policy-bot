from backend.config import csv_env


def test_csv_env_trims_spaces_empty_items_and_trailing_slashes(monkeypatch):
    monkeypatch.setenv("CORS_TEST", " https://policy-bot.vercel.app/ , ,http://localhost:3000")
    assert csv_env("CORS_TEST", "") == ["https://policy-bot.vercel.app", "http://localhost:3000"]


def test_csv_env_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("CORS_TEST", raising=False)
    assert csv_env("CORS_TEST", "a,b") == ["a", "b"]