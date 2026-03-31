from __future__ import annotations

from argparse import Namespace
from _pytest.monkeypatch import MonkeyPatch

from healingstone.api import cli


def test_cli_main_routes_to_service(monkeypatch: MonkeyPatch) -> None:
    seen = {"parse_called": False, "service_called": False}

    def fake_parse_args() -> Namespace:
        seen["parse_called"] = True
        return Namespace(config="configs/pipeline.yaml")

    def fake_execute(args: Namespace) -> None:
        assert args.config == "configs/pipeline.yaml"
        seen["service_called"] = True

    monkeypatch.setattr(cli, "parse_args", fake_parse_args)
    monkeypatch.setattr(cli, "execute_reconstruction", fake_execute)

    cli.main()

    assert seen == {"parse_called": True, "service_called": True}
