from zq_rag_app.core.config import Settings


def test_neo4j_settings_are_loaded_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://graph.example:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "graph-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "graph-password")
    monkeypatch.setenv("NEO4J_DATABASE", "knowledge")

    configured = Settings(
        _env_file=None,
        dashscope_api_key="test-dashscope-key",
        reranker_endpoint="https://example.invalid/rerank",
        jwt_secret_key="test-jwt-secret",
    )

    assert configured.neo4j_uri == "bolt://graph.example:7687"
    assert configured.neo4j_username == "graph-user"
    assert configured.neo4j_password == "graph-password"
    assert configured.neo4j_database == "knowledge"
