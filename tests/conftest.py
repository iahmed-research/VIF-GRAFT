from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

# Windows managed workspaces can reject pytest's owner-only temporary ACL.
# Keep isolated artifacts under the ignored local directory using normal ACLs.
import uuid
import pytest
@pytest.fixture
def tmp_path():
    path=Path(__file__).resolve().parents[1]/'local/tests'/uuid.uuid4().hex
    path.mkdir(parents=True)
    return path
