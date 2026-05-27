from htc_heom_fast.run_htc import parse_args


def test_cli_parser_defaults(monkeypatch):
    monkeypatch.setattr('sys.argv', ['htc-heom-run'])
    args = parse_args()
    assert args.Nmol == 5
    assert args.K_matsubara == 0
    assert args.store == 'obs'
