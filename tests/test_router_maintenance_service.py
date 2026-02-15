"""Módulo `tests/test_router_maintenance_service.py`: descreve responsabilidades e integrações deste arquivo."""

from app.services.router_maintenance import create_background_threads


def test_create_background_threads():
    """Testa create background threads."""
    def _f():
        """Executa f."""
        return None

    ts = create_background_threads(_f, _f, _f, _f, _f)
    assert len(ts) == 5
    assert all(t.daemon for t in ts)
    assert ts[0].name == "router-cleanup-query-log"
