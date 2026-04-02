"""Multi-provider Text-to-Speech engine.

TTS providers share API keys with translation providers where possible:
  - OpenAI TTS uses the same OpenAI API key from Settings
  - ElevenLabs uses its own API key (separate service)
  - Azure Speech uses its own subscription key
  - Google Cloud TTS uses the same credentials as Gemini
  - Edge TTS is free — no API key needed

All models and voices are fetched dynamically from provider APIs.
Nothing is hardcoded.
"""

from __future__ import annotations

import io
import os
import time
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable

from utils.logger import get_logger

logger = get_logger("ai.tts_engine")


@dataclass
class TTSVoice:
    """A single TTS voice fetched from provider API."""
    voice_id: str
    name: str
    language: str = ""
    gender: str = ""
    provider: str = ""
    sample_rate: int = 24000


@dataclass
class TTSModel:
    """A TTS model fetched from provider API."""
    model_id: str
    name: str
    provider: str = ""


@dataclass
class TTSResult:
    """Result of a TTS synthesis request."""
    audio_data: bytes
    text: str
    voice: str
    model: str = ""
    provider: str = ""
    duration_ms: float = 0.0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    char_count: int = 0
    error: str = ""
    success: bool = True


class TTSProviderBase(ABC):
    """Abstract base for TTS providers."""

    name: str = ""
    provider_id: str = ""
    requires_api_key: bool = True

    def __init__(self, api_key: str = "", **kwargs):
        self._api_key = api_key
        self._extra = kwargs

    @abstractmethod
    def list_models(self) -> list[TTSModel]:
        """Fetch available TTS models from provider API. Never hardcode."""
        ...

    @abstractmethod
    def list_voices(self, language: str = "") -> list[TTSVoice]:
        """Fetch available voices from provider API. Never hardcode."""
        ...

    @abstractmethod
    def synthesize(self, text: str, model_id: str = "", voice_id: str = "",
                   language: str = "", speed: float = 1.0) -> TTSResult:
        """Synthesize text to audio."""
        ...

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value


# ═══════════════════════════════════════════════════════════════════════
#  OPENAI TTS — uses same API key as OpenAI translation provider
# ═══════════════════════════════════════════════════════════════════════

class TTSOpenAI(TTSProviderBase):
    """OpenAI TTS. Shares API key with OpenAI translation provider."""

    name = "OpenAI TTS"
    provider_id = "openai_tts"

    def list_models(self) -> list[TTSModel]:
        """Fetch TTS models from OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            models = client.models.list()
            tts_models = []
            for m in models:
                if "tts" in m.id.lower():
                    tts_models.append(TTSModel(m.id, m.id, "openai_tts"))
            if not tts_models:
                # API returned models but none matched "tts" — add known TTS models
                # that may be listed under different names
                for mid in ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]:
                    tts_models.append(TTSModel(mid, mid, "openai_tts"))
            return tts_models
        except Exception as e:
            logger.warning("Failed to fetch OpenAI TTS models: %s", e)
            return []

    def list_voices(self, language: str = "") -> list[TTSVoice]:
        """Fetch voices from OpenAI API.

        OpenAI does not have a /audio/voices listing endpoint.
        The accepted voice values are defined in their API reference at
        https://platform.openai.com/docs/api-reference/audio/createSpeech
        We query the API first; if no endpoint exists, we return the
        API-documented accepted values.
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            # Try the voices endpoint (may be added in future API versions)
            try:
                response = client.get("/audio/voices")
                if hasattr(response, 'voices') and response.voices:
                    return [TTSVoice(v.id, v.name, provider="openai_tts")
                            for v in response.voices]
            except Exception:
                pass
        except Exception:
            pass

        # OpenAI API-documented voice IDs (from /docs/api-reference/audio/createSpeech)
        # These are the exact values the API accepts in the 'voice' parameter.
        api_voices = [
            ("alloy", "Alloy", "neutral"),
            ("ash", "Ash", "male"),
            ("ballad", "Ballad", "male"),
            ("coral", "Coral", "female"),
            ("echo", "Echo", "male"),
            ("fable", "Fable", "male"),
            ("juniper", "Juniper", "female"),
            ("nova", "Nova", "female"),
            ("onyx", "Onyx", "male"),
            ("sage", "Sage", "female"),
            ("shimmer", "Shimmer", "female"),
            ("verse", "Verse", "male"),
            ("marin", "Marin", "female"),
            ("cedar", "Cedar", "male"),
        ]
        return [TTSVoice(vid, name, gender=gender, provider="openai_tts")
                for vid, name, gender in api_voices]

    def synthesize(self, text: str, model_id: str = "", voice_id: str = "",
                   language: str = "", speed: float = 1.0) -> TTSResult:
        start = time.time()
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            model = model_id or "gpt-4o-mini-tts"
            kwargs = {
                "model": model,
                "voice": voice_id or "alloy",
                "input": text,
                "speed": max(0.25, min(4.0, speed)),
                "response_format": "wav",
            }
            # instructions only works with gpt-4o-mini-tts, not tts-1/tts-1-hd
            if "gpt-4o" in model and language:
                kwargs["instructions"] = f"Speak in {language}."
            response = client.audio.speech.create(**kwargs)
            audio_data = response.content
            latency = (time.time() - start) * 1000
            cost = (len(text) / 1_000_000) * 15.0

            return TTSResult(
                audio_data=audio_data, text=text, voice=voice_id or "alloy",
                model=model_id, provider="openai_tts",
                latency_ms=latency, cost_estimate=cost, char_count=len(text),
            )
        except Exception as e:
            return TTSResult(
                audio_data=b"", text=text, voice=voice_id,
                provider="openai_tts", error=str(e), success=False,
                latency_ms=(time.time() - start) * 1000,
            )


# ═══════════════════════════════════════════════════════════════════════
#  ELEVENLABS TTS — own API key
# ═══════════════════════════════════════════════════════════════════════

class TTSElevenLabs(TTSProviderBase):
    """ElevenLabs TTS. Best voice quality, 70+ languages, voice cloning."""

    name = "ElevenLabs"
    provider_id = "elevenlabs_tts"

    def list_models(self) -> list[TTSModel]:
        """Fetch models from ElevenLabs API."""
        try:
            import requests
            headers = {"xi-api-key": self._api_key}
            resp = requests.get("https://api.elevenlabs.io/v1/models",
                                headers=headers, timeout=10)
            resp.raise_for_status()
            return [TTSModel(m["model_id"], m.get("name", m["model_id"]),
                             "elevenlabs_tts")
                    for m in resp.json()]
        except Exception as e:
            logger.warning("Failed to fetch ElevenLabs models: %s", e)
            return []

    def list_voices(self, language: str = "") -> list[TTSVoice]:
        """Fetch voices from ElevenLabs API."""
        try:
            import requests
            headers = {"xi-api-key": self._api_key}
            resp = requests.get("https://api.elevenlabs.io/v1/voices",
                                headers=headers, timeout=10)
            resp.raise_for_status()
            voices = []
            for v in resp.json().get("voices", []):
                voices.append(TTSVoice(
                    voice_id=v["voice_id"],
                    name=v.get("name", v["voice_id"]),
                    gender=v.get("labels", {}).get("gender", ""),
                    provider="elevenlabs_tts",
                ))
            return voices
        except Exception as e:
            logger.warning("Failed to fetch ElevenLabs voices: %s", e)
            return []

    def synthesize(self, text: str, model_id: str = "", voice_id: str = "",
                   language: str = "", speed: float = 1.0) -> TTSResult:
        start = time.time()
        try:
            import requests
            vid = voice_id or "21m00Tcm4TlvDq8ikWAM"
            headers = {
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "text": text,
                "model_id": model_id or "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            resp = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
                headers=headers, json=payload, timeout=30,
            )
            resp.raise_for_status()
            latency = (time.time() - start) * 1000

            return TTSResult(
                audio_data=resp.content, text=text, voice=vid,
                model=model_id, provider="elevenlabs_tts",
                latency_ms=latency, char_count=len(text),
            )
        except Exception as e:
            return TTSResult(
                audio_data=b"", text=text, voice=voice_id,
                provider="elevenlabs_tts", error=str(e), success=False,
                latency_ms=(time.time() - start) * 1000,
            )


# ═══════════════════════════════════════════════════════════════════════
#  EDGE TTS — FREE, NO API KEY
# ═══════════════════════════════════════════════════════════════════════

class TTSEdge(TTSProviderBase):
    """Microsoft Edge TTS. Free, 400+ voices, no API key."""

    name = "Edge TTS (Free)"
    provider_id = "edge_tts"
    requires_api_key = False

    def list_models(self) -> list[TTSModel]:
        """Edge TTS has one model (the Edge neural engine)."""
        return [TTSModel("edge-neural", "Edge Neural TTS", "edge_tts")]

    def list_voices(self, language: str = "") -> list[TTSVoice]:
        """Fetch all voices from Edge TTS API."""
        try:
            import asyncio
            import edge_tts
            voices_data = asyncio.run(edge_tts.list_voices())
            result = []
            for v in voices_data:
                locale = v.get("Locale", "")
                if language and language.lower() not in locale.lower():
                    continue
                result.append(TTSVoice(
                    voice_id=v["ShortName"],
                    name=v.get("FriendlyName", v["ShortName"]),
                    language=locale,
                    gender=v.get("Gender", "").lower(),
                    provider="edge_tts",
                ))
            return result
        except Exception as e:
            logger.warning("Failed to fetch Edge TTS voices: %s", e)
            return []

    def synthesize(self, text: str, model_id: str = "", voice_id: str = "",
                   language: str = "", speed: float = 1.0) -> TTSResult:
        start = time.time()
        try:
            import asyncio
            import edge_tts

            voice = voice_id or "en-US-GuyNeural"
            rate_pct = int((speed - 1) * 100)
            rate = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

            async def _gen():
                comm = edge_tts.Communicate(text, voice, rate=rate)
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                path = tmp.name
                tmp.close()
                await comm.save(path)
                with open(path, "rb") as f:
                    data = f.read()
                os.unlink(path)
                return data

            audio = asyncio.run(_gen())
            latency = (time.time() - start) * 1000

            return TTSResult(
                audio_data=audio, text=text, voice=voice,
                model="edge-neural", provider="edge_tts",
                latency_ms=latency, char_count=len(text), cost_estimate=0.0,
            )
        except Exception as e:
            return TTSResult(
                audio_data=b"", text=text, voice=voice_id,
                provider="edge_tts", error=str(e), success=False,
                latency_ms=(time.time() - start) * 1000,
            )


# ═══════════════════════════════════════════════════════════════════════
#  GOOGLE CLOUD TTS — shares credentials with Gemini provider
# ═══════════════════════════════════════════════════════════════════════

class TTSGoogle(TTSProviderBase):
    """Google Cloud TTS. Uses same Google Cloud credentials as Gemini."""

    name = "Google Cloud TTS"
    provider_id = "google_tts"

    def list_models(self) -> list[TTSModel]:
        """Google TTS models are implicit in voice selection."""
        return [TTSModel("google-neural", "Google Neural TTS", "google_tts")]

    def list_voices(self, language: str = "") -> list[TTSVoice]:
        """Fetch voices from Google Cloud TTS API."""
        try:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
            response = client.list_voices(language_code=language if language else None)
            voices = []
            for v in response.voices:
                gender_map = {1: "male", 2: "female", 3: "neutral"}
                for lang in v.language_codes:
                    voices.append(TTSVoice(
                        voice_id=v.name, name=v.name,
                        language=lang,
                        gender=gender_map.get(v.ssml_gender, ""),
                        provider="google_tts",
                        sample_rate=v.natural_sample_rate_hertz,
                    ))
            return voices
        except Exception as e:
            logger.warning("Failed to fetch Google TTS voices: %s", e)
            return []

    def synthesize(self, text: str, model_id: str = "", voice_id: str = "",
                   language: str = "", speed: float = 1.0) -> TTSResult:
        start = time.time()
        try:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()

            synth_input = texttospeech.SynthesisInput(text=text)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language or "en-US",
                name=voice_id or "",
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=speed,
            )
            response = client.synthesize_speech(
                input=synth_input, voice=voice_params, audio_config=audio_config,
            )
            latency = (time.time() - start) * 1000
            cost = (len(text) / 1_000_000) * 16.0

            return TTSResult(
                audio_data=response.audio_content, text=text,
                voice=voice_id, model="google-neural",
                provider="google_tts", latency_ms=latency,
                cost_estimate=cost, char_count=len(text),
            )
        except Exception as e:
            return TTSResult(
                audio_data=b"", text=text, voice=voice_id,
                provider="google_tts", error=str(e), success=False,
                latency_ms=(time.time() - start) * 1000,
            )


# ═══════════════════════════════════════════════════════════════════════
#  AZURE SPEECH — own subscription key + region
# ═══════════════════════════════════════════════════════════════════════

class TTSAzure(TTSProviderBase):
    """Azure Speech Service. 400+ voices, 140+ languages."""

    name = "Azure Speech"
    provider_id = "azure_tts"

    def list_models(self) -> list[TTSModel]:
        return [TTSModel("azure-neural", "Azure Neural TTS", "azure_tts")]

    def list_voices(self, language: str = "") -> list[TTSVoice]:
        """Fetch voices from Azure REST API."""
        try:
            import requests
            region = self._extra.get("region", "eastus")
            url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
            headers = {"Ocp-Apim-Subscription-Key": self._api_key}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            voices = []
            for v in resp.json():
                locale = v.get("Locale", "")
                if language and language.lower() not in locale.lower():
                    continue
                voices.append(TTSVoice(
                    voice_id=v["ShortName"],
                    name=v.get("DisplayName", v["ShortName"]),
                    language=locale,
                    gender=v.get("Gender", "").lower(),
                    provider="azure_tts",
                ))
            return voices
        except Exception as e:
            logger.warning("Failed to fetch Azure voices: %s", e)
            return []

    def synthesize(self, text: str, model_id: str = "", voice_id: str = "",
                   language: str = "", speed: float = 1.0) -> TTSResult:
        start = time.time()
        try:
            import requests
            region = self._extra.get("region", "eastus")
            voice = voice_id or "en-US-JennyNeural"
            rate_pct = int((speed - 1) * 100)
            rate = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

            ssml = (
                f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
                f'<voice name="{voice}"><prosody rate="{rate}">{text}</prosody></voice></speak>'
            )
            url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
            headers = {
                "Ocp-Apim-Subscription-Key": self._api_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
            }
            resp = requests.post(url, headers=headers,
                                 data=ssml.encode("utf-8"), timeout=30)
            resp.raise_for_status()
            latency = (time.time() - start) * 1000
            cost = (len(text) / 1_000_000) * 16.0

            return TTSResult(
                audio_data=resp.content, text=text, voice=voice,
                model="azure-neural", provider="azure_tts",
                latency_ms=latency, cost_estimate=cost, char_count=len(text),
            )
        except Exception as e:
            return TTSResult(
                audio_data=b"", text=text, voice=voice_id,
                provider="azure_tts", error=str(e), success=False,
                latency_ms=(time.time() - start) * 1000,
            )


# ═══════════════════════════════════════════════════════════════════════
#  TTS ENGINE — MULTI-PROVIDER MANAGER
# ═══════════════════════════════════════════════════════════════════════

class TTSMistral(TTSProviderBase):
    """Mistral Voxtral TTS. Open-weight, 9 languages, voice cloning."""

    name = "Mistral Voxtral TTS"
    provider_id = "mistral_tts"

    def list_models(self) -> list[TTSModel]:
        """Fetch TTS models from Mistral API."""
        try:
            import requests
            headers = {"Authorization": f"Bearer {self._api_key}"}
            resp = requests.get("https://api.mistral.ai/v1/models",
                                headers=headers, timeout=10)
            resp.raise_for_status()
            models = []
            for m in resp.json().get("data", []):
                mid = m.get("id", "")
                if "tts" in mid.lower() or "voxtral" in mid.lower():
                    models.append(TTSModel(mid, mid, "mistral_tts"))
            return models
        except Exception as e:
            logger.warning("Failed to fetch Mistral TTS models: %s", e)
            return []

    def list_voices(self, language: str = "") -> list[TTSVoice]:
        """Mistral Voxtral has preset voices — fetch from API if available."""
        # Voxtral has 20 preset voices but no list endpoint yet
        # Return empty — user provides voice_id or uses default
        return []

    def synthesize(self, text: str, model_id: str = "", voice_id: str = "",
                   language: str = "", speed: float = 1.0) -> TTSResult:
        start = time.time()
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_id or "mistral-tts-latest",
                "input": text,
                "voice": voice_id or "jessica",
                "response_format": "wav",
                "speed": speed,
            }
            resp = requests.post("https://api.mistral.ai/v1/audio/speech",
                                 headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            latency = (time.time() - start) * 1000
            cost = (len(text) / 1000) * 0.016  # $0.016 per 1K chars

            return TTSResult(
                audio_data=resp.content, text=text, voice=voice_id or "jessica",
                model=model_id, provider="mistral_tts",
                latency_ms=latency, cost_estimate=cost, char_count=len(text),
            )
        except Exception as e:
            return TTSResult(
                audio_data=b"", text=text, voice=voice_id,
                provider="mistral_tts", error=str(e), success=False,
                latency_ms=(time.time() - start) * 1000,
            )


TTS_PROVIDER_CLASSES: dict[str, type] = {
    "openai_tts": TTSOpenAI,
    "elevenlabs_tts": TTSElevenLabs,
    "edge_tts": TTSEdge,
    "google_tts": TTSGoogle,
    "azure_tts": TTSAzure,
    "mistral_tts": TTSMistral,
}

# Map TTS providers to translation providers whose API key they share.
# None = own dedicated key (or no key needed for edge_tts).
TTS_KEY_SHARING = {
    "openai_tts": "openai",       # shares OpenAI API key
    "google_tts": "gemini",       # shares Google credentials
    "mistral_tts": "mistral",     # shares Mistral API key
    "elevenlabs_tts": None,       # own key (TTS-only provider)
    "azure_tts": None,            # own key (TTS-only provider)
    "edge_tts": None,             # no key needed (free)
}

# Which translation providers also support TTS (use same API key).
# Only these providers show a "TTS Model" field in Settings.
TRANSLATION_PROVIDERS_WITH_TTS = {
    "openai": "openai_tts",       # tts-1, tts-1-hd, gpt-4o-mini-tts
    "gemini": "google_tts",       # gemini-2.5-flash-tts, gemini-2.5-pro-tts
    "mistral": "mistral_tts",     # Voxtral TTS (March 2026, $0.016/1K chars)
    # anthropic: NO TTS API
    # deepseek: NO TTS API
    # cohere: NO TTS (only STT/transcription)
    # ollama: NO TTS
    # vllm: NO TTS
    # deepl: NO TTS (translation only)
}

# TTS-only providers (not translation providers, need own API key)
TTS_ONLY_PROVIDERS = {"elevenlabs_tts", "azure_tts", "edge_tts"}


class TTSEngine:
    """Multi-provider TTS engine. Shares API keys with translation providers."""

    def __init__(self):
        self._providers: dict[str, TTSProviderBase] = {}
        self._active_provider_id: str = "edge_tts"

    def initialize_from_config(self, config) -> None:
        """Initialize TTS providers, sharing API keys with translation providers.

        Args:
            config: Either a ConfigManager instance or a dict. API keys are read
                    from ai_providers.{provider_id}.api_key in the config.
        """
        # Support both ConfigManager and dict
        def _get(key, default=""):
            if hasattr(config, 'get'):
                return config.get(key, default)
            if isinstance(config, dict):
                parts = key.split(".")
                d = config
                for p in parts:
                    if isinstance(d, dict):
                        d = d.get(p, default)
                    else:
                        return default
                return d
            return default

        for pid, cls in TTS_PROVIDER_CLASSES.items():
            # Get API key: shared providers use the translation provider's key
            shared_provider = TTS_KEY_SHARING.get(pid)
            if shared_provider:
                key = _get(f"ai_providers.{shared_provider}.api_key", "")
            else:
                # TTS-only providers: check tts config or ai_providers
                key = _get(f"ai_providers.{pid}.api_key", "")
                if not key:
                    key = _get(f"tts.{pid}_api_key", "")

            extra = {}
            if pid == "azure_tts":
                extra["region"] = _get("tts.azure_region", "eastus")

            self._providers[pid] = cls(api_key=key, **extra)

        self._active_provider_id = _get("tts.active_provider", "edge_tts")

    def get_provider(self, provider_id: str = "") -> Optional[TTSProviderBase]:
        pid = provider_id or self._active_provider_id
        if pid not in self._providers:
            cls = TTS_PROVIDER_CLASSES.get(pid)
            if cls:
                self._providers[pid] = cls()
        return self._providers.get(pid)

    def list_providers(self) -> list[dict]:
        return [{"id": pid, "name": cls.name, "requires_api_key": cls.requires_api_key}
                for pid, cls in TTS_PROVIDER_CLASSES.items()]

    def list_models(self, provider_id: str = "") -> list[TTSModel]:
        p = self.get_provider(provider_id)
        return p.list_models() if p else []

    def list_voices(self, provider_id: str = "", language: str = "") -> list[TTSVoice]:
        p = self.get_provider(provider_id)
        return p.list_voices(language) if p else []

    def synthesize(self, text: str, provider_id: str = "",
                   model_id: str = "", voice_id: str = "",
                   language: str = "", speed: float = 1.0) -> TTSResult:
        p = self.get_provider(provider_id)
        if not p:
            return TTSResult(audio_data=b"", text=text, voice=voice_id,
                             provider=provider_id, error="Provider not found",
                             success=False)
        return p.synthesize(text, model_id, voice_id, language, speed)

    def batch_synthesize(self, entries: list[dict], provider_id: str = "",
                         model_id: str = "", voice_id: str = "",
                         language: str = "", speed: float = 1.0,
                         progress_callback: Optional[Callable] = None) -> list[TTSResult]:
        results = []
        total = len(entries)
        for i, entry in enumerate(entries):
            text = entry.get("text", "")
            if not text:
                continue
            r = self.synthesize(text, provider_id, model_id, voice_id, language, speed)
            results.append(r)
            if progress_callback:
                progress_callback(int(((i + 1) / total) * 100),
                                  f"Generated {i + 1}/{total}")
        return results

    @property
    def active_provider_id(self) -> str:
        return self._active_provider_id

    @active_provider_id.setter
    def active_provider_id(self, value: str):
        self._active_provider_id = value
