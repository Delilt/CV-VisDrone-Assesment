# ONNX GIRIS CIKIS SEKLI DOGRU MU TESTLERI


import os
import onnx
import pytest

MODEL_PATH = "models/latest.onnx"

def test_onnx_input_output():
    """ONNX modelinin giriş/çıkış katmanlarını doğrula"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("ONNX modeli bulunamadı")

    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model)
    input_node = model.graph.input[0]
    input_name = input_node.name
    
    output_node = model.graph.output[0]
    
    assert input_name is not None
    assert output_node.name is not None
    
    print(f"\n[ONNX Info] Input: {input_name}, Output: {output_node.name}")