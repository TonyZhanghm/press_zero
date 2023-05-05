from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/persist/hzhang/data/mercury_7B_sft_hf"

print("Model Loading...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer Loaded!")
model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
model.eval()
print("Model Loaded!")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    # get input data
    input_data = request.get_json()["input"]
    model_input = tokenizer(input_data, return_tensors="pt").input_ids.cuda()
    print("Start generating for input: {}".format(input_data))
    
    with torch.no_grad():
        result = model.generate(model_input, max_length=256)
    output = tokenizer.decode(result[0], skip_special_tokens=True)
    print(output)
    return jsonify({'predictions': "here's the input: \n{}\nand here's the output: \n {}".format(input_data, output)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)