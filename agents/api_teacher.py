"""
API-based teacher fallback client.
Tries Groq first (fast, free-tier), falls back to Gemini if Groq fails/rate-limits.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class APITeacher:
    def __init__(self, groq_model: str = "llama-3.3-70b-versatile", gemini_model: str = "gemini-2.0-flash"):
        self.groq_model = groq_model
        self.gemini_model = gemini_model

        self._groq_client = None
        self._gemini_client = None

        if GROQ_API_KEY:
            from groq import Groq
            self._groq_client = Groq(api_key=GROQ_API_KEY)

        if GEMINI_API_KEY:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self._gemini_client = genai.GenerativeModel(self.gemini_model)

    def _call_groq(self, prompt: str, max_tokens: int = 1024) -> str:
        if not self._groq_client:
            raise RuntimeError("Groq client not configured (missing GROQ_API_KEY)")
        response = self._groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        if not self._gemini_client:
            raise RuntimeError("Gemini client not configured (missing GEMINI_API_KEY)")
        response = self._gemini_client.generate_content(prompt)
        return response.text

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Try Groq first. On any failure (rate limit, error, missing key),
        fall back to Gemini. Raises if both fail.
        """
        try:
            return self._call_groq(prompt, max_tokens=max_tokens)
        except Exception as groq_err:
            print(f"[api_teacher] Groq failed ({groq_err}), falling back to Gemini...")
            try:
                return self._call_gemini(prompt)
            except Exception as gemini_err:
                raise RuntimeError(
                    f"Both teachers failed. Groq error: {groq_err} | Gemini error: {gemini_err}"
                )


if __name__ == "__main__":
    teacher = APITeacher()
    result = teacher.generate("Write a Python function that reverses a string.")
    print(result)