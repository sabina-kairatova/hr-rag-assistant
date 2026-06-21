import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_chat_endpoint_basic():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.post("/chat", json={"message": "hello", "thread_id": "test-1"})

    assert response.status_code == 200
    data = response.json()

    assert "response" in data
    assert "thread_id" in data
    assert "model_used" in data
    assert "cached" in data
    assert "processing_time_ms" in data
    assert "timestamp" in data
    assert isinstance(data["response"], str)


@pytest.mark.asyncio
async def test_chat_endpoint_with_context_retrieveal():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.post(
                "/chat", json={"message": "Does a company offer any retirement plan?", "thread_id": "test-2"}
            )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["response"], str)
    words_to_find = ["offers", "retirement plan", "401(k)", "eligible", "save", "contributions", "earnings"]
    assert any(word in data["response"] for word in words_to_find)


@pytest.mark.asyncio
async def test_chat_endpoint_with_keyword_retrieveal():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.post(
                "/chat", json={"message": "What reasons may I take the FMLA leave for?", "thread_id": "test-3"}
            )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["response"], str)
    words_to_find = ["birth", "care", "child", "health", "spouse", "parent"]
    assert any(word in data["response"] for word in words_to_find)
    

@pytest.mark.asyncio
async def test_chat_endpoint_with_non_hr_query():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.post(
                "/chat", json={"message": "What is 2+2?", "thread_id": "test-4"}
            )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["response"], str)
    words_to_find = ["sorry", "unable", "unfortunately", "not related to HR", "HR-related"]
    assert any(word in data["response"] for word in words_to_find)


@pytest.mark.asyncio
async def test_chat_endpoint_validation_error():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "environment" in data
    assert "version" in data
    assert "checks" in data


@pytest.mark.asyncio
async def test_metrics_endpoint():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_errors" in data
    assert "error_rate" in data
    assert "avg_latency_ms" in data
    assert "cache_hit_rate" in data
    assert "total_input_tokens" in data
    assert "total_output_tokens" in data


@pytest.mark.asyncio
async def test_cache_stats_endpoint():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "hits" in data
    assert "misses" in data
    assert "hit_rate" in data
    assert "cached_entries" in data