#!/usr/bin/env python3
"""
Simple HTTP embedding server using sentence-transformers.
Provides semantic embeddings for the VPK demo.
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

app = Flask(__name__)

# Try GPU, fall back to CPU if CUDA isn't usable on this machine.
def _select_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # Probe with a tiny tensor — catches driver/kernel mismatches that
        # torch.cuda.is_available() misses (e.g. no kernel image for device).
        t = torch.zeros(1, device="cuda")
        del t
        return "cuda"
    except Exception:
        return "cpu"

DEVICE = _select_device()

print(f"Loading sentence-transformers model (device={DEVICE})...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'all-MiniLM-L6-v2',
        'device': DEVICE,
        'dimension': model.get_sentence_embedding_dimension()
    })

@app.route('/embed', methods=['POST'])
def embed():
    """
    Embed a single text.

    Request body:
    {
        "text": "patient with chest pain and fatigue"
    }

    Response:
    {
        "embedding": [0.123, -0.456, ...],
        "dimension": 384
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate embedding and normalize to unit length
        embedding = model.encode(text, normalize_embeddings=True)

        return jsonify({
            'embedding': embedding.tolist(),
            'dimension': len(embedding)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/embed_batch', methods=['POST'])
def embed_batch():
    """
    Embed multiple texts in batch.

    Request body:
    {
        "texts": ["text1", "text2", "text3"]
    }

    Response:
    {
        "embeddings": [[0.123, ...], [0.456, ...], ...],
        "count": 3,
        "dimension": 384
    }
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])

        if not texts:
            return jsonify({'error': 'No texts provided'}), 400

        # Generate embeddings and normalize to unit length
        embeddings = model.encode(texts, normalize_embeddings=True)

        return jsonify({
            'embeddings': embeddings.tolist(),
            'count': len(embeddings),
            'dimension': embeddings.shape[1] if len(embeddings) > 0 else 0
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 Embedding Server Starting")
    print("="*60)
    print(f"Model:     all-MiniLM-L6-v2")
    print(f"Device:    {DEVICE}")
    print(f"Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"Endpoint:  http://localhost:5001")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=False)
