import io

import pandas as pd
import requests

from kikolens.config import ollama_generate_url, ollama_model, ollama_timeout


def get_ai_insights(df: pd.DataFrame, ollama_api_base_url: str = None, model_name: str = None) -> str:
    """Genera insights en inglés con el modelo local de Ollama."""
    url = ollama_api_base_url or ollama_generate_url()
    model = model_name or ollama_model()

    buffer = io.StringIO()
    df.info(buf=buffer)

    prompt = (
        "Role: Expert Data Analyst. Task: Analyze this dataset and provide 3 HIGH-IMPACT insights. "
        "Language: English. Format: Professional Markdown bullet points.\n\n"
        f"DATA SAMPLE:\n{df.head(3).to_string()}\n\n"
        f"STRUCTURE:\n{buffer.getvalue()}"
    )

    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=ollama_timeout(),
        )
        response.raise_for_status()
        return response.json().get("response", "AI: Analysis completed.")
    except requests.exceptions.Timeout:
        return "❌ **AI Timeout:** The local model is taking too long. Try a lighter model like llama3.2."
    except Exception as e:
        return f"⚠️ **Error:** {e}"
