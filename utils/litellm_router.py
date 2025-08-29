from litellm import Router

def create_router(
    gemini_api_key: str = "",
    openai_api_key: str = "",
    deepseek_api_key: str = "",
    openrouter_api_key: str = "",
) -> Router:
    """Create a LiteLLM Router with advanced load balancing and fallback configuration."""
    model_list = [
        {
            "model_name": "image-classifier",
            "litellm_params": {
                "model": "openrouter/google/gemini-2.5-flash", # 16k tokens output
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "text-classifier",
            "litellm_params": {
                "model": "openrouter/openai/gpt-4o-mini",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 20,  # Higher for text-only tasks
                "weight": 1,
            }
        },
        {
            "model_name": "html-splitter",
            "litellm_params": {
                "model": "openrouter/x-ai/grok-3-mini", # 16k tokens output
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "html-splitter",
            "litellm_params": {
                "model": "openai/gpt-4o-mini", # 16k tokens output
                "api_key": openai_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "html-splitter",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini", # 32k tokens output
                "api_key": openai_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "openrouter/anthropic/claude-sonnet-4", # 128k tokens output
                "api_key": openrouter_api_key,
                "max_parallel_requests": 3,
                "weight": 3,
            }
        },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "openrouter/openai/gpt-oss-120b", # 133k tokens output
                "api_key": openrouter_api_key,
                "max_parallel_requests": 3,
                "weight": 1,
            }
        },
        {
            "model_name": "html-parser",
            "litellm_params": {
                "model": "openrouter/qwen/qwen3-235b-a22b-thinking-2507", # 131k tokens output
                "api_key": openrouter_api_key,
                "max_parallel_requests": 3,
                "weight": 1,
            }
        },
        {
            "model_name": "structure-detector",
            "litellm_params": {
                "model": "gemini/gemini-2.5-flash", # openrouter-hosted version fails on this task!
                "api_key": gemini_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "page-mapper",
            "litellm_params": {
                "model": "openrouter/deepseek/deepseek-chat",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 3,
            }
        },
        {
            "model_name": "page-mapper",
            "litellm_params": {
                "model": "openrouter/openai/gpt-4o-mini",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        },
        {
            "model_name": "page-mapper",
            "litellm_params": {
                "model": "openrouter/openai/gpt-4.1-mini",
                "api_key": openrouter_api_key,
                "max_parallel_requests": 10,
                "weight": 1,
            }
        }
    ]

    # Router configuration
    return Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",  # Weighted random selection
        fallbacks=[
            {"image-classifier": ["image-classifier"]},
            {"html-splitter": ["html-splitter"]},
            {"html-parser": ["html-parser"]},
            {"structure-detector": ["structure-detector"]},
            {"page-mapper": ["page-mapper"]},
            {"text-classifier": ["page-mapper", "html-parser"]}
        ],  # Falls back within the same group
        num_retries=2,
        allowed_fails=5,
        cooldown_time=30,
        enable_pre_call_checks=True,  # Enable context window and rate limit checks
        default_max_parallel_requests=50,  # Global default
        set_verbose=False,  # Set to True for debugging
    )