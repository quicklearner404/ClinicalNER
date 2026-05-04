from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

model_id = "./models/biomed_ner"
save_dir = "./models/biomed_ner_onnx"

# Export to ONNX
model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("✅ Optimization Complete: ONNX model saved to models/biomed_ner_onnx")