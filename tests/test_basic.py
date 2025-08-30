from training.train_model import load_data, clean_text
def test_data_loaded():
    X, y, df = load_data()
    assert len(df) > 0
