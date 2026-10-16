import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from logging import Logger
from typing import Protocol, TypedDict, cast, runtime_checkable
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.llm_result import LLMResult

GEMINI_PRO_INPUT_COST_PER_1K_TOKENS = 0.0001
GEMINI_PRO_OUTPUT_COST_PER_1K_TOKENS = 0.0004

#per_k
# storage per hour
class ModelCost(TypedDict):
    input: float
    output: float
    cache: float
    storage_per_hour: float


class UsageHistoryEntry(TypedDict):
    model_name: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost: float
    start_time: float | None
    end_time: float | None


class TokenDetails(TypedDict, total=False):
    cache_read: int | float | str
    reasoning: int | float | str


class UsageMetadata(TypedDict, total=False):
    input_tokens: int | float | str
    output_tokens: int | float | str
    input_token_details: TokenDetails
    output_token_details: TokenDetails


@runtime_checkable
class _UsageMetadataCarrier(Protocol):
    @property
    def usage_metadata(self) -> object: ...


@runtime_checkable
class _GenerationInfoCarrier(Protocol):
    @property
    def generation_info(self) -> object: ...


@runtime_checkable
class _MessageCarrier(Protocol):
    @property
    def message(self) -> object: ...


@runtime_checkable
class _ResponseMetadataCarrier(Protocol):
    @property
    def response_metadata(self) -> object: ...


cost_table: dict[str, ModelCost] = {
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004, "cache" : 0.0, "storage_per_hour": 0.0},
    "gemini-1.5-pro": {"input": 0.001250, "output": 0.005, "cache" : 0.0, "storage_per_hour": 0.0},
    "gemini-2.5-flash-preview-04-17": {"input": 0.000150, "output": 0.0035, "cache" : 0.0000375, "storage_per_hour": 0.0010},
    "gemini-2.5-flash": {"input": 0.000300, "output": 0.0025, "cache" : 0.0000375, "storage_per_hour": 0.0010},
    "gemini-2.5-pro-preview-03-25": {"input": 0.001250, "output": 0.0100, "cache": 0.00031, "storage_per_hour": 0.0045},
    "gemini-2.5-pro": {"input": 0.001250, "output": 0.0100, "cache": 0.00031, "storage_per_hour": 0.0045},
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0040, "cache": 0.00025, "storage_per_hour": 0.001}
}


def _usage_metadata(message: object) -> UsageMetadata | None:
    value = message.usage_metadata if isinstance(message, _UsageMetadataCarrier) else None
    return cast(UsageMetadata, dict(value)) if isinstance(value, Mapping) else None


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(value))
        except ValueError:
            return 0
    return 0


def _usage_count(usage: Mapping[str, object], key: str) -> int:
    return _non_negative_int(usage.get(key, 0))


def _nested_usage_count(usage: Mapping[str, object], key: str, nested_key: str) -> int:
    details = usage.get(key)
    if not isinstance(details, Mapping):
        return 0
    return _non_negative_int(details.get(nested_key, 0))


def _model_name(generation: object) -> str:
    value = (
        generation.generation_info
        if isinstance(generation, _GenerationInfoCarrier)
        else None
    )
    if isinstance(value, Mapping):
        model = value.get("model_name")
        if isinstance(model, str):
            return model
    return "unknown"


def _generation_message(generation: object) -> object | None:
    if isinstance(generation, _MessageCarrier):
        return generation.message
    return None


def _response_metadata(message: object) -> object | None:
    if isinstance(message, _ResponseMetadataCarrier):
        return message.response_metadata
    return None
keys = list(cost_table.keys())
for k in keys:
    if k.startswith('models/'):
        pass
    else:
        cost_table['models/' + k] = cost_table[k]

def calculate_gemini_cost(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    model_name: str,
) -> float:
    """Calculates the cost based on Gemini Pro pricing."""
    cost = cost_table.get(
        model_name,
        {
            "input": 1.250 / 1000000,
            "output": 5.0 / 1000000,
            "cache": 0.0,
            "storage_per_hour": 0.0,
        },
    )
    input_cost = ((input_tokens-cached_tokens) / 1000) * cost['input']
    cache_cost = cached_tokens / 1000 * cost['cache']
    output_cost = (output_tokens / 1000) * cost['output']
    total_cost = input_cost + output_cost + cache_cost
    return total_cost
class GeminiCostCallbackHandler(BaseCallbackHandler):
    """A custom callback handler to track Gemini API costs."""
    
    def __init__(self) -> None:
        super().__init__()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_tokens = 0
        self.reasoning_tokens = 0
        self.total_cost = 0.0
        self.usage_history: list[UsageHistoryEntry] = []
        self.run_start_time: float | None = None
        self.run_end_time: float | None = None
    def on_llm_start(
        self,
        serialized: dict[str, object],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        **kwargs: object,
    ) -> object:
        self.run_start_time = time.time()
        return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        """Called at the end of an LLM call."""
        self.run_end_time = time.time()
        for generation in response.generations:
            # The 'generation' is a list of ChatGeneration or Generation objects
            for gen in generation:
                # Check if the generation object is a ChatGeneration instance
                # and has the 'usage_metadata' attribute.
                if isinstance(gen, ChatGeneration) and hasattr(gen, 'message'):
                    message = gen.message
                    usage_metadata = _usage_metadata(message)
                    if usage_metadata is not None:
                                    
                    
                            input_tokens = _usage_count(usage_metadata, "input_tokens")
                            output_tokens = _usage_count(usage_metadata, "output_tokens")
                            cached_tokens = _nested_usage_count(
                                usage_metadata, "input_token_details", "cache_read"
                            )
                            reasoning_tokens = _nested_usage_count(
                                usage_metadata, "output_token_details", "reasoning"
                            )

                            if input_tokens > 0 or output_tokens > 0:
                                cost = calculate_gemini_cost(input_tokens, output_tokens, cached_tokens, model_name=_model_name(gen))
                                self.total_input_tokens += input_tokens
                                self.total_output_tokens += output_tokens
                                self.reasoning_tokens += reasoning_tokens
                                self.cache_tokens += cached_tokens
                                self.total_cost += cost
                                self.usage_history.append({
                                    "model_name": _model_name(gen),
                                    "input_tokens" : input_tokens,
                                    "output_tokens":output_tokens,
                                    "cached_tokens":cached_tokens,
                                    "reasoning_tokens":reasoning_tokens,
                                    "cost":cost,
                                    "start_time": self.run_start_time,
                                    "end_time" : self.run_end_time

                                })

    def reset(self) -> None:
        """Resets the counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
    def __repr__(self) -> str:
        return (
            f"Total Input Tokens: {self.total_input_tokens}\n"
            f"Total Output Tokens: {self.total_output_tokens}\n"
            f"Total Cost: ${self.total_cost:.8f}"
        )
    def model_dump(self) -> dict[str, object]:
        return (
            {"input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            'usage_history' : self.usage_history}
        )
@contextmanager
def get_gemini_callback_cost() -> Iterator[GeminiCostCallbackHandler]:
    """A context manager to track Gemini API costs for a block of code."""
    # Create an instance of the handler
    """_summary_

    Yields:
        _type_: _description_
        
    _usage_
    with get_gemini_callback_cost() as cb:
        result = chain.invoke(
            {"city": "Paris"},
            config={"callbacks": [cb]} # Pass the yielded handler to the chain
        )
        print("\n--- Inside Context Manager ---")
        print(result.content)

        # You can access the cost immediately after the call
        print("\n--- Cost After First Call ---")
        print(cb)

    # The state is preserved even after the `with` block exits
    print("\n--- Final Cost from Context Manager ---")
    print(cb)
        
    """
    callback_handler = GeminiCostCallbackHandler()
    try:
        # Yield the handler so it can be used inside the 'with' block
        # and its state can be accessed after the block.
        yield callback_handler
    finally:
        # The code inside the 'with' block has finished.
        # The handler now holds the final cost.
        pass

class PromptCostTokenLogger(BaseCallbackHandler):
    def __init__(self, logger: Logger) -> None:
        self.cost_token_logger: Logger = logger
        self.total_input_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cached_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: object,
    ) -> object:
        # self.cost_token_logger.info(response.response_metadata)
        for g in response.generations:
            for gg in g:
                
                to_log: dict[str, str | None] = {'response_metadata': None, 'usage_metadata': None}
                message = _generation_message(gg)
                response_metadata = (
                    _response_metadata(message) if message is not None else None
                )
                if response_metadata is not None:
                    to_log["response_metadata"] = f"{response_metadata}"
                usage = _usage_metadata(message) if message is not None else None
                if usage is not None:
                    to_log["usage_metadata"] = f"{usage}"
                if to_log:
                    self.cost_token_logger.info(str(to_log))
# example "{'input_tokens': 106170, 'output_tokens': 7652, 'total_tokens': 118594, 'input_token_details': {'cache_read': 106164}, 'output_token_details': {'reasoning': 4772}}"
        for generation in response.generations:
            # The 'generation' is a list of ChatGeneration or Generation objects
            for gen in generation:
                # Check if the generation object is a ChatGeneration instance
                # and has the 'usage_metadata' attribute.
                message = _generation_message(gen)
                if message is not None and _usage_metadata(message) is not None:
                    
                    usage_metadata = _usage_metadata(message)
                    if usage_metadata is None:
                        continue
                    
                    input_tokens = _usage_count(usage_metadata, "input_tokens")
                    cached_tokens = _nested_usage_count(
                        usage_metadata, "input_token_details", "cache_read"
                    )
                    output_tokens = _usage_count(usage_metadata, "output_tokens")
                    reasoning_tokens = _nested_usage_count(
                        usage_metadata, "output_token_details", "reasoning"
                    )
                    if input_tokens > 0 or output_tokens > 0:
                        cost = calculate_gemini_cost(input_tokens, output_tokens, cached_tokens, model_name=_model_name(gen))
                        self.total_input_tokens += input_tokens
                        self.total_cached_tokens += cached_tokens
                        self.total_output_tokens += output_tokens
                        self.total_reasoning_tokens += reasoning_tokens
                        self.total_cost += cost
        return super().on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: object,
    ) -> object:
        return super().on_llm_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )
class PromptTokenCounter(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict[str, object],
        prompts: list[str],
        **kwargs: object,
    ) -> None:
        for prompt in prompts:
            token_count = self.count_tokens(prompt)
            print(f"Prompt: {prompt}")
            print(f"Token count: {token_count}")

    def count_tokens(self, text: str) -> int:
        # Implement your token counting logic here
        # For example, a simple approximation:
        return len(text.split())
