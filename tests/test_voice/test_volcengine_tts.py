from __future__ import annotations

import base64

from tau2.data_model.voice import VolcengineTTSConfig
from tau2.voice.utils.volcengine_tts_utils import tts_volcengine


def test_doubao_tts_api_key_takes_precedence(monkeypatch):
    monkeypatch.delenv("VOLCENGINE_TTS_API_KEY", raising=False)
    monkeypatch.setenv("DOUBAO_TTS_API_KEY", "dedicated-tts-key")
    monkeypatch.setenv("DOUBAO_SEED_API_KEY", "other-seed-key")

    class Response:
        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode):
            assert decode_unicode is True
            yield '{"code": 0, "data": "' + base64.b64encode(b"audio").decode() + '"}'
            yield '{"code": 20000000}'

    seen = {}

    def fake_post(url, *, headers, **kwargs):
        seen["url"] = url
        seen["headers"] = headers
        return Response()

    monkeypatch.setattr(
        "tau2.voice.utils.volcengine_tts_utils.requests.post", fake_post
    )
    result = tts_volcengine("hello", VolcengineTTSConfig())

    assert seen["headers"]["X-Api-Key"] == "dedicated-tts-key"
    assert seen["headers"]["X-Api-Resource-Id"] == "seed-tts-2.0"
    assert result.data == b"audio"
