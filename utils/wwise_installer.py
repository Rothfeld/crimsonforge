"""Wwise auto-detection and WAV→WEM Vorbis conversion.

Finds WwiseConsole.exe from:
  1. WWISEROOT environment variable
  2. Program Files / Program Files (x86) scan
  3. User-specified path in settings

Uses WwiseConsole.exe convert-external-source to produce proper
Vorbis-encoded WEM files that game engines accept.

If Wwise is not installed, guides the user to install it (free).
"""

import os
import glob
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

from utils.logger import get_logger

logger = get_logger("utils.wwise")


def find_wwise_console() -> str:
    """Find WwiseConsole.exe on this system.

    Search order:
      1. WWISEROOT environment variable
      2. Common install locations (Program Files)
      3. All drives, Wwise folders

    Returns:
        Path to WwiseConsole.exe or empty string if not found.
    """
    # 1. Check WWISEROOT env
    wwiseroot = os.environ.get("WWISEROOT", "")
    if wwiseroot:
        console = os.path.join(wwiseroot, "Authoring", "x64", "Release", "bin",
                               "WwiseConsole.exe")
        if os.path.isfile(console):
            return console
        # Also check directly
        console = os.path.join(wwiseroot, "WwiseConsole.exe")
        if os.path.isfile(console):
            return console

    # 2. Check common locations
    search_roots = []
    for env in ["ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"]:
        p = os.environ.get(env, "")
        if p:
            search_roots.append(p)

    # Also check user AppData (Wwise Launcher installs here)
    appdata = os.environ.get("LOCALAPPDATA", "")
    if appdata:
        search_roots.append(os.path.join(appdata, "Audiokinetic"))

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        # Search for Wwise folders
        for d in os.listdir(root):
            if "wwise" in d.lower() or "audiokinetic" in d.lower():
                wwise_dir = os.path.join(root, d)
                # Search recursively for WwiseConsole.exe
                for dirpath, dirnames, filenames in os.walk(wwise_dir):
                    for fn in filenames:
                        if fn.lower() == "wwiseconsole.exe":
                            return os.path.join(dirpath, fn)

    # 3. Check PATH
    path = shutil.which("WwiseConsole")
    if path:
        return path

    return ""


def is_wwise_installed() -> bool:
    """Check if Wwise is installed."""
    return bool(find_wwise_console())


def convert_wav_to_wem_vorbis(
    wav_path: str,
    output_path: str = "",
    sample_rate: int = 48000,
    channels: int = 1,
    quality: str = "4",
    wwise_console: str = "",
) -> str:
    """Convert WAV to Vorbis-encoded WEM using WwiseConsole.exe.

    This produces proper Wwise Vorbis WEM files that game engines accept.

    Args:
        wav_path: Input WAV file path.
        output_path: Output WEM path. Auto-generated if empty.
        sample_rate: Target sample rate (default 48000).
        channels: Target channels (default 1 = mono).
        quality: Vorbis quality (0-10, default "4" = medium).
        wwise_console: Path to WwiseConsole.exe. Auto-detected if empty.

    Returns:
        Path to output WEM file, or empty string on failure.
    """
    if not wwise_console:
        wwise_console = find_wwise_console()
    if not wwise_console:
        logger.error("WwiseConsole.exe not found. Install Wwise from audiokinetic.com")
        return ""

    if not output_path:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        output_path = os.path.join(tempfile.gettempdir(), f"cf_wem_{basename}.wem")

    # First normalize WAV with ffmpeg if available
    from core.audio_converter import get_ffmpeg_path
    ffmpeg = get_ffmpeg_path()
    normalized_wav = wav_path
    if ffmpeg:
        norm_path = wav_path + ".norm.wav"
        try:
            result = subprocess.run(
                [ffmpeg, "-y", "-i", wav_path,
                 "-ar", str(sample_rate), "-ac", str(channels),
                 "-sample_fmt", "s16", norm_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and os.path.isfile(norm_path):
                normalized_wav = norm_path
        except Exception:
            pass

    # Create temporary Wwise project structure
    tmp_dir = tempfile.mkdtemp(prefix="cf_wwise_")
    try:
        # Create minimal .wproj
        wproj_path = os.path.join(tmp_dir, "cf_convert.wproj")
        _write_minimal_wproj(wproj_path, quality)

        # Create .wsources XML listing the WAV file
        wsources_path = os.path.join(tmp_dir, "convert.wsources")
        output_dir = os.path.dirname(output_path) or tempfile.gettempdir()
        _write_wsources(wsources_path, normalized_wav, output_dir, quality)

        # Run WwiseConsole
        cmd = [
            wwise_console,
            "convert-external-source",
            wproj_path,
            "--source-file", wsources_path,
            "--output", output_dir,
            "--quiet",
        ]
        logger.info("Running WwiseConsole: %s", " ".join(cmd))

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            logger.error("WwiseConsole failed (rc=%d): %s", result.returncode, result.stderr)
            # Try alternative approach: direct convert
            cmd2 = [
                wwise_console,
                "convert-external-source",
                wproj_path,
                "--source-file", wsources_path,
                "--output", output_dir,
            ]
            result = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)

        # Find the output WEM
        basename = os.path.splitext(os.path.basename(normalized_wav))[0]
        possible_outputs = [
            os.path.join(output_dir, f"{basename}.wem"),
            os.path.join(output_dir, "Windows", f"{basename}.wem"),
            os.path.join(output_dir, "Windows", "SFX", f"{basename}.wem"),
        ]

        # Also search recursively
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".wem"):
                    possible_outputs.append(os.path.join(root, f))

        for candidate in possible_outputs:
            if os.path.isfile(candidate) and os.path.getsize(candidate) > 100:
                if candidate != output_path:
                    shutil.copy2(candidate, output_path)
                logger.info("Wwise converted: %s -> %s (%d bytes)",
                            wav_path, output_path, os.path.getsize(output_path))
                return output_path

        logger.error("WwiseConsole produced no output. stdout: %s", result.stdout[:500])
        return ""

    finally:
        # Cleanup temp project
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        # Cleanup normalized wav
        if normalized_wav != wav_path and os.path.isfile(normalized_wav):
            try:
                os.unlink(normalized_wav)
            except Exception:
                pass


def _write_minimal_wproj(path: str, quality: str = "4"):
    """Write a minimal Wwise project file for conversion."""
    content = f"""<?xml version="1.0" encoding="utf-8"?>
<WwiseDocument Type="WorkUnit" SchemaVersion="110">
    <ProjectInfo>
        <Project Name="cf_convert" Version="1">
            <Property Name="DefaultConversion" Type="string" Value="Vorbis Quality {quality}"/>
        </Project>
    </ProjectInfo>
</WwiseDocument>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_wsources(path: str, wav_path: str, output_dir: str, quality: str = "4"):
    """Write a .wsources XML file listing WAV files for conversion."""
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    abs_wav = os.path.abspath(wav_path)

    content = f"""<?xml version="1.0" encoding="utf-8"?>
<ExternalSourcesList SchemaVersion="1" Root="{os.path.dirname(abs_wav)}">
    <Source Path="{os.path.basename(abs_wav)}" Conversion="Vorbis Quality {quality}">
        <Comment>{basename}</Comment>
    </Source>
</ExternalSourcesList>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
