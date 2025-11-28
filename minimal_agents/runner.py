class RunnerResult:
    def __init__(self, final_output: str):
        self.final_output = final_output

class Runner:
    @staticmethod
    async def run(agent, input: str, run_config):
        provider = run_config.model_provider
        response = await provider.complete(
            instructions=agent.instructions,
            input_text=input
        )
        return RunnerResult(response)
