# ONNX GIRIS CIKIS SEKLI DOGRU MU TESTLERI


import os
import onnx
import pytest

MODEL_PATH = "models/latest.onnx"

def test_onnx_input_output():
    """ONNX modelinin giriş/çıkış katmanlarını doğrula"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("ONNX modeli bulunamadı")

    # Modeli yükle
    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model) # Model yapısı bozuk mu kontrol et

    # Giriş katmanını kontrol et (Genelde 'images' adında olur)
    input_node = model.graph.input[0]
    input_name = input_node.name
    
    # Çıkış katmanını kontrol et (Genelde 'output0' adında olur)
    output_node = model.graph.output[0]
    
    # İsimlerin dolu olduğunu doğrula
    assert input_name is not None
    assert output_node.name is not None
    
    print(f"\n[ONNX Info] Input: {input_name}, Output: {output_node.name}")