from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from inference_onnx import OnnxInference

app = FastAPI(title="MLOps Practice - Paraphrase Detection ONNX Inference API")

predictor = OnnxInference("./models/mrpc_model.onnx")


@app.get("/predict/")
async def predict(sentence1: str, sentence2: str):
    """
    Predict if two sentences are paraphrases of each other.

    - **sentence1**: First sentence
    - **sentence2**: Second sentence
    """
    result = predictor.predict(sentence1, sentence2)
    predicted_label = max(result, key=lambda x: x["score"])
    return {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "predicted_label": predicted_label["label"],
        "confidence": predicted_label["score"],
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Paraphrase Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            textarea {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
                margin-top: 10px;
            }
            button:hover {
                background-color: #0056b3;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 4px;
                display: none;
            }
            .paraphrase {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .not-paraphrase {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .confidence {
                font-weight: bold;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Paraphrase Detection</h1>
            <p style="text-align: center; color: #666;">Check if two sentences are paraphrases of each other</p>
            
            <label for="sentence1"><strong>Sentence 1:</strong></label>
            <textarea id="sentence1" rows="3" placeholder="Enter first sentence..."></textarea>
            
            <label for="sentence2"><strong>Sentence 2:</strong></label>
            <textarea id="sentence2" rows="3" placeholder="Enter second sentence..."></textarea>
            
            <button onclick="checkParaphrase()">Check Paraphrase</button>
            
            <div id="result"></div>
        </div>

        <script>
            async function checkParaphrase() {
                const sentence1 = document.getElementById('sentence1').value;
                const sentence2 = document.getElementById('sentence2').value;
                
                if (!sentence1 || !sentence2) {
                    alert('Please enter both sentences');
                    return;
                }
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = 'Loading...';
                resultDiv.style.display = 'block';
                resultDiv.className = '';
                
                try {
                    const response = await fetch(
                        `/predict/?sentence1=${encodeURIComponent(sentence1)}&sentence2=${encodeURIComponent(sentence2)}`
                    );
                    const data = await response.json();
                    
                    const isParaphrase = data.predicted_label === 'paraphrase';
                    const confidence = (data.confidence * 100).toFixed(2);
                    
                    resultDiv.className = isParaphrase ? 'paraphrase' : 'not-paraphrase';
                    resultDiv.innerHTML = `
                        <strong>Result:</strong> ${isParaphrase ? '✓ Paraphrase' : '✗ Not a Paraphrase'}<br>
                        <div class="confidence">Confidence: ${confidence}%</div>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = 'Error: ' + error.message;
                    resultDiv.style.backgroundColor = '#fff3cd';
                }
            }
        </script>
    </body>
    </html>
    """
