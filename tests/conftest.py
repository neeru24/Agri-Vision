@pytest.fixture(scope="session", autouse=True)
def mock_models():
    """Session-wide fixture to mock the heavy ML models."""
    app_module.resnet_model = MockResNetModel()
    app_module.yolo_model = MockYOLOModel()
    app_module.model_manager.resnet_model = app_module.resnet_model
    app_module.model_manager.yolo_model = app_module.yolo_model
    app_module.model_manager.loaded = True
    yield
    # Restore original attributes if needed (though session is ending)
    if hasattr(app_module.model_manager, 'resnet_model'):
        app_module.resnet_model = app_module.model_manager.resnet_model
    if hasattr(app_module.model_manager, 'yolo_model'):
        app_module.yolo_model = app_module.model_manager.yolo_model