"""Error handling: LLM response validation with retry logic + exponential backoff."""

import time

from pydantic import BaseModel, ValidationError
from src.llm.utils import parse_json_response
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3


def validate_llm_response(
    llm_client,
    prompt: str,
    system_prompt: str,
    schema: type[BaseModel],
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Call the LLM and validate the response against a Pydantic schema.

    On validation failure, sends the error back to the LLM as correction feedback.
    Returns the validated response as a dict.
    Raises ValueError after max_retries failed attempts.
    """
    current_prompt = prompt
    last_error = None

    for attempt in range(1, max_retries + 1):
        response = llm_client.complete(current_prompt, system_prompt=system_prompt)
        raw = parse_json_response(response)

        try:
            validated = schema.model_validate(raw)
            return validated.model_dump()
        except ValidationError as e:
            last_error = e
            error_details = _format_validation_errors(e)
            logger.warning("Validation failed (attempt %d/%d): %s", attempt, max_retries, error_details)

            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info("Backoff: waiting %ds before retry", wait)
                time.sleep(wait)

            current_prompt = (
                f"Ta reponse precedente contient des erreurs de validation :\n"
                f"{error_details}\n\n"
                f"Corrige ta reponse en respectant strictement le schema demande.\n\n"
                f"Rappel de la demande originale :\n{prompt}"
            )

    raise ValueError(
        f"LLM response failed validation after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def _format_validation_errors(error: ValidationError) -> str:
    """Format Pydantic validation errors into a readable string for the LLM."""
    parts = []
    for e in error.errors():
        loc = " -> ".join(str(x) for x in e["loc"])
        parts.append(f"- Champ '{loc}': {e['msg']} (valeur recue: {e.get('input', '?')})")
    return "\n".join(parts)
