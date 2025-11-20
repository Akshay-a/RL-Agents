"""
Tests for SegmentPro API endpoints
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import create_tables


@pytest.fixture(scope="module")
def client():
    """Create test client"""
    create_tables()
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def test_user(client):
    """Create and return a test user with API key"""
    # Register user
    response = client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    assert response.status_code == 200
    user = response.json()

    # Get API keys
    # Login first to get token
    response = client.post(
        "/auth/login",
        params={"email": "test@example.com", "password": "testpassword123"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    # Get API keys
    response = client.get(
        "/api-keys",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    api_keys = response.json()

    return {
        "user": user,
        "token": token,
        "api_key": api_keys[0]["key"] if api_keys else None
    }


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test health endpoint returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data


class TestAuthEndpoints:
    """Test authentication endpoints"""

    def test_register_user(self, client):
        """Test user registration"""
        response = client.post(
            "/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "full_name": "New User"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["tier"] == "free"

    def test_register_duplicate_email(self, client):
        """Test registering with duplicate email fails"""
        # First registration
        client.post(
            "/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "password123"
            }
        )

        # Second registration with same email
        response = client.post(
            "/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "password456"
            }
        )
        assert response.status_code == 400

    def test_login_valid(self, client, test_user):
        """Test login with valid credentials"""
        response = client.post(
            "/auth/login",
            params={"email": "test@example.com", "password": "testpassword123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid(self, client):
        """Test login with invalid credentials"""
        response = client.post(
            "/auth/login",
            params={"email": "test@example.com", "password": "wrongpassword"}
        )
        assert response.status_code == 401


class TestSegmentationEndpoints:
    """Test segmentation endpoints"""

    def test_segment_image_unauthorized(self, client):
        """Test segmentation without auth fails"""
        response = client.post(
            "/segment/image",
            json={
                "prompt": "test object"
            }
        )
        assert response.status_code == 401

    def test_segment_image_with_api_key(self, client, test_user):
        """Test segmentation with API key"""
        response = client.post(
            "/segment/image",
            headers={"X-API-Key": test_user["api_key"]},
            json={
                "prompt": "main object",
                "prompt_type": "text"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "completed"
        assert "masks" in data

    def test_segment_image_with_token(self, client, test_user):
        """Test segmentation with JWT token"""
        response = client.post(
            "/segment/image",
            headers={"Authorization": f"Bearer {test_user['token']}"},
            json={
                "prompt": "product",
                "prompt_type": "text"
            }
        )
        assert response.status_code == 200


class TestUsageEndpoints:
    """Test usage and billing endpoints"""

    def test_get_usage(self, client, test_user):
        """Test getting usage statistics"""
        response = client.get(
            "/usage",
            headers={"X-API-Key": test_user["api_key"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "billing_period" in data
        assert "api_calls_used" in data
        assert "current_tier" in data

    def test_get_pricing(self, client):
        """Test getting pricing tiers"""
        response = client.get("/pricing")
        assert response.status_code == 200
        data = response.json()
        assert "free" in data
        assert "starter" in data
        assert "pro" in data


class TestModelEndpoints:
    """Test model info endpoints"""

    def test_get_model_info(self, client):
        """Test getting model information"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "SAM 3"
        assert "capabilities" in data
