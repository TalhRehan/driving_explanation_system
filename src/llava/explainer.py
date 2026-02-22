"""
LLaVA inference wrapper.

The model is loaded once on first use and reused across calls.
generate() must only be called when delta_exec == 1; the pipeline
enforces this, but an internal guard is included as a safeguard.

Heavy imports (torch, transformers) are deferred until the first
inference call so the rest of the pipeline can be imported and tested
without these packages installed.
"""

from pathlib import Path
from typing import Optional


class LLaVAExplainer:
    """
    Thin wrapper around LLaVA-1.5.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. 'llava-hf/llava-1.5-7b-hf'.
    max_new_tokens : int
        Maximum tokens in the generated explanation.
    device : str | None
        'cuda', 'cpu', or None (auto-detected).
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        max_new_tokens: int = 256,
        device: Optional[str] = None,
    ):
        self._model_name    = model_name
        self._max_new_tokens = max_new_tokens
        self._device_pref   = device   # resolved lazily
        self._model         = None
        self._processor     = None
        self._device        = None

    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy load — deferred until the first actual inference call."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers and torch are required for LLaVA inference.\n"
                "Run: pip install transformers torch Pillow"
            )

        self._device = self._device_pref or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading LLaVA model '{self._model_name}' on {self._device} ...")

        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------

    def generate(
        self,
        image_path: str,
        action: str,
        context: dict,
        evidence: list[dict],
        delta_exec: int,
        query: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate a natural-language explanation for the current driving moment.

        Parameters
        ----------
        image_path : str
            Path to the RGB frame.
        action : str
            Discrete driving action label (e.g. 'brake', 'yield').
        context : dict
            Compact context: speed, ttc_t, crit_zone.
        evidence : list[dict]
            Bounding box records from ground-truth or YOLO detection.
        delta_exec : int
            Must be 1; returns None immediately if 0.
        query : str | None
            Optional user query; falls back to default prompt template.

        Returns
        -------
        str | None
            Generated explanation text, or None if delta_exec == 0.
        """
        if delta_exec != 1:
            return None

        import torch
        from PIL import Image as PILImage

        self._load()

        image  = PILImage.open(image_path).convert("RGB")
        prompt = self._build_prompt(action, context, evidence, query)

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        action: str,
        context: dict,
        evidence: list[dict],
        query: Optional[str],
    ) -> str:
        speed_kmh = round(context.get("speed", 0.0) * 3.6, 1)
        ttc       = context.get("ttc_t", "N/A")
        crit      = "yes" if context.get("crit_zone") else "no"

        evidence_lines = [
            f"  - {e.get('class', 'object')} at {e.get('distance', '?')} m"
            for e in evidence[:5]
        ]
        evidence_str = "\n".join(evidence_lines) if evidence_lines else "  - none detected"

        task = query or (
            f"Explain why the vehicle chose to {action} given the scene. "
            "Be concise and refer to specific objects visible in the image."
        )

        return (
            f"USER: <image>\n"
            f"The vehicle is travelling at {speed_kmh} km/h. "
            f"Time-to-collision: {ttc} s. "
            f"Near intersection: {crit}.\n"
            f"Nearby objects:\n{evidence_str}\n\n"
            f"{task}\nASSISTANT:"
        )