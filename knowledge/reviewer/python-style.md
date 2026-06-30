# Team Python Style Guide (sample knowledge for the reviewer)

These are the house rules the code reviewer persona should enforce.

Naming:
- Functions and variables use snake_case. Classes use PascalCase.
- Constants are UPPER_SNAKE_CASE and defined at module top.

Structure:
- Keep functions under 40 lines. If longer, split out a helper.
- Prefer pure functions; isolate side effects (I/O, network) at the edges.
- One module = one responsibility.

Specific house rules (use these to test retrieval):
- Never use a bare `except:`. Always catch a specific exception type.
- All public functions must have type hints on arguments and return value.
- Secrets always load from environment variables, never hardcoded.
- Maximum line length is 100 characters, not 79.
