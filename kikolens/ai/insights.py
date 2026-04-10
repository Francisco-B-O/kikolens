import os
import pandas as pd
import requests
import io

def get_ai_insights(df: pd.DataFrame, ollama_api_base_url: str = None, model_name: str = "llama3.1") -> str:
    """
    Generates AI-powered insights in English with a 60s timeout.
    """
    if ollama_api_base_url is None:
        ollama_api_base_url = f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/generate"
    df_sample = df.head(3).to_string() 
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()

    # Prompt forced to ENGLISH
    prompt = (
        "Role: Expert Data Analyst. Task: Analyze this dataset and provide 3 HIGH-IMPACT insights. "
        "Language: English. Format: Professional Markdown bullet points.\n\n"
        f"DATA SAMPLE:\n{df_sample}\n\n"
        f"STRUCTURE:\n{df_info_str}"
    )

    data = {
        "model": model_name, "prompt": prompt, "stream": False
    }

    try:
        response = requests.post(ollama_api_base_url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "AI: Analysis completed.")
    except requests.exceptions.Timeout:
        return "❌ **AI Timeout:** The local model is taking too long. Try a lighter model like llama3.2."
    except Exception as e:
        return f"⚠️ **Error:** {e}"
