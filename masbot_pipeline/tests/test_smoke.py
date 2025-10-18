def test_smoke_imports():
    import importlib
    modules = [
        'src.io_utils',
        'src.detect_tags',
        'src.track_pipeline',
        'src.metrics',
        'src.plotting',
        'src.calibrate',
        'src.main',
    ]
    for m in modules:
        importlib.import_module(m)


