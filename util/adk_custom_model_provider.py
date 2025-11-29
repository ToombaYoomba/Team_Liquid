class CustomModelProvider:
    def __init__(self, model, client):
        self.model = model
        self.client = client

    async def complete(self, instructions: str, input_text: str) -> str:
        resp = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text
        )
        return resp.output_text
