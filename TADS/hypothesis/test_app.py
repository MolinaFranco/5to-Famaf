from app import app
from hypothesis import given, strategies as st

def client():
    app.testing = True
    return app.test_client()

@given(a=st.integers(), b=st.integers())
def test_sum_endpoint_with_valid_ints(a, b):
    response = client().get(f"/sum?a={a}&b={b}")
    assert response.status_code == 200
    data = response.get_json()
    assert data["result"] == a + b

@given(text=st.text())
def test_sum_endpoint_with_invalid_input(text):
    # Testea entrada inválida en 'a' o 'b'
    response = client().get(f"/sum?a={text}&b=1")
    if not text.isdigit():
        assert response.status_code == 400
