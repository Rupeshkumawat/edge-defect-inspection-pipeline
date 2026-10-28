import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

input_fp32 = "models/model_fp32.onnx"
output_int8 = "models/model_int8.onnx"

print(f"Quantizing {input_fp32} -> {output_int8} ...")
quantize_dynamic(
    model_input=input_fp32,
    model_output=output_int8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=True
)
print("Quantization complete!")
