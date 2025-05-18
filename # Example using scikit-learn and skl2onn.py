# Example using scikit-learn and skl2onnx
# (Assuming 'model' is your trained scikit-learn model and 'initial_types' describes input)
# import skl2onnx
# onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_types)
# with open("scheduler_model.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())