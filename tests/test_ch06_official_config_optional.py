"""Optional online entrypoint contract checked with fakes, never Hub access."""
import json

from examples import ch06_official_config_optional as module


def test_optional_entry_downloads_only_pinned_config(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'config.json'
    path.write_text('{}')
    calls = []

    def fake_download(model, filename, revision):
        calls.append((model, filename, revision))
        return str(path)

    class FakeConfig:
        text_config = object()

    monkeypatch.setattr(module, 'hf_hub_download', fake_download)
    monkeypatch.setattr(module.Qwen3_5Config, 'from_dict', lambda value: FakeConfig())

    def fake_inspect(config, *, meta):
        assert config is FakeConfig.text_config and meta is True
        return None, {'kind': 'fake_not_model_measurement'}

    monkeypatch.setattr(module, 'inspect_model', fake_inspect)
    module.main()
    assert calls == [(module.MODEL, 'config.json', module.REVISION)]
    report = json.loads(capsys.readouterr().out)
    assert 'vision tower excluded' in report['scope']
    assert report['config_revision'] == module.REVISION
