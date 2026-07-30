class LLMAsJudge:
    def __init__(self, model_name: str = "llama3-70b"):
        self.model_name = model_name

    def evaluate(self, prompt: str, response: str, target: str) -> dict:
        score = 5 if response.strip() == target.strip() else 1
        return {"score": score, "reason": "Vergleich basiert auf exaktem Text-Match."}
