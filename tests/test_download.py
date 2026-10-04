import hashlib
import io
from collections import Counter
from threading import Barrier, Lock
import pytest
from graft import data


def fixture_server(monkeypatch):
    rows=[{'record':'r','event':f'e{i}'} for i in range(4)]
    files={name:b'metadata\n' for name in ('LICENSE.txt','benchmark_data_split.csv','event_labels.csv')}
    files.update({f'waveforms/r/e{i}.hea':b'r 1 250 10\nshared.dat 16 1/mV 16 0 0 0 0 II\n' for i in range(4)})
    files['waveforms/r/shared.dat']=b'waveform bytes'
    files['SHA256SUMS.txt']=''.join(hashlib.sha256(v).hexdigest()+'  '+k+'\n' for k,v in files.items()).encode()
    counts=Counter();lock=Lock();barrier=Barrier(4)
    def urlopen(url,timeout):
        name=url.removeprefix(data.URL)
        with lock:counts[name]+=1
        # All four requests must start concurrently, or this test fails.
        if name.endswith('.hea'):barrier.wait(timeout=10)
        return io.BytesIO(files[name])
    monkeypatch.setattr(data.urllib.request,'urlopen',urlopen)
    return rows,files,counts


def test_parallel_download_shared_files_and_resume(tmp_path,monkeypatch):
    rows,files,counts=fixture_server(monkeypatch)
    data.download(tmp_path,rows)
    assert counts['waveforms/r/shared.dat']==1
    assert all((tmp_path/k).read_bytes()==v for k,v in files.items())
    prior=counts.copy()
    data.download(tmp_path,rows)
    assert counts==prior
    (tmp_path/'waveforms/r/shared.dat').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Existing data checksum differs'):
        data.download(tmp_path,rows)


def test_download_rejects_corrupt_response(tmp_path,monkeypatch):
    rows,files,counts=fixture_server(monkeypatch)
    files['waveforms/r/shared.dat']=b'wrong response'
    with pytest.raises(ValueError,match='Downloaded checksum differs'):
        data.download(tmp_path,rows)
    assert not (tmp_path/'waveforms/r/shared.dat').exists()
