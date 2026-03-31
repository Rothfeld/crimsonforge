"""Deep trace: Localization Key -> UI Binding Pipeline.

Uses the app's own VFS to decrypt and analyze real game .html, .thtml,
.css, .xml files — traces how paloc keys connect to the game UI.
Writes output to a file to avoid Windows console encoding issues.
"""

import os
import sys
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vfs_manager import VfsManager
from core.paloc_parser import parse_paloc
from utils.platform_utils import auto_discover_game

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_output.txt")


def run():
    f = open(OUT_FILE, "w", encoding="utf-8")
    def P(s=""):
        f.write(s + "\n")
        f.flush()

    pkg = auto_discover_game()
    if not pkg:
        P("ERROR: Game not found.")
        f.close()
        return

    P(f"Game: {pkg}")
    vfs = VfsManager(pkg)
    vfs.load_papgt()
    groups = vfs.list_package_groups()
    P(f"Groups: {len(groups)}")

    total = 0
    for g in groups:
        try:
            pamt = vfs.load_pamt(g)
            total += len(pamt.file_entries)
        except:
            pass
    P(f"Total files: {total:,}\n")

    # Categorize
    ui_files = defaultdict(list)
    for g, pamt in vfs._pamt_cache.items():
        for entry in pamt.file_entries:
            ext = os.path.splitext(entry.path.lower())[1]
            if ext in (".html", ".thtml", ".css", ".xml", ".json",
                       ".uianiminit", ".pami", ".paloc", ".txt"):
                ui_files[ext].append((g, entry))

    P("=" * 70)
    P("  FILE COUNTS")
    P("=" * 70)
    for ext in sorted(ui_files.keys()):
        P(f"  {ext:15s} : {len(ui_files[ext]):,}")

    # ── PHASE 1: ALL HTML/THTML ──
    P(f"\n{'='*70}")
    P("  PHASE 1: ALL HTML/THTML files (decrypted)")
    P("=" * 70)

    html_entries = ui_files.get(".html", []) + ui_files.get(".thtml", [])
    P(f"Total: {len(html_entries)}")

    loc_patterns = defaultdict(int)

    for i, (g, entry) in enumerate(html_entries):
        try:
            data = vfs.read_entry_data(entry)
            text = data.decode("utf-8", errors="replace")

            P(f"\n{'_'*60}")
            P(f"[{i+1}/{len(html_entries)}] {entry.path}")
            P(f"Group: {g} | Size: {len(data):,} bytes")
            P("_" * 60)
            P(text[:4000])
            if len(text) > 4000:
                P(f"... ({len(text)-4000:,} more chars)")

            for m in re.finditer(r'(data-\w+)\s*=\s*"([^"]*)"', text):
                loc_patterns[m.group(1)] += 1
            for m in re.finditer(r'(\w+)\s*\(\s*["\']([^"\']{3,})["\']', text):
                fn = m.group(1)
                if len(fn) > 2 and fn[0].islower():
                    loc_patterns[f"fn:{fn}()"] += 1
            for m in re.finditer(r'\{\{([^}]+)\}\}', text):
                loc_patterns[f"tmpl:{{{{{m.group(1)[:30]}}}}}"] += 1
            for m in re.finditer(r'(\w+)\s*=\s*"(\d{5,})"', text):
                loc_patterns[f"numattr:{m.group(1)}"] += 1

        except Exception as e:
            P(f"  ERROR: {e}")

    # ── PHASE 2: ALL CSS ──
    P(f"\n{'='*70}")
    P("  PHASE 2: ALL CSS files (decrypted)")
    P("=" * 70)

    css_entries = ui_files.get(".css", [])
    P(f"Total: {len(css_entries)}")

    for i, (g, entry) in enumerate(css_entries):
        try:
            data = vfs.read_entry_data(entry)
            text = data.decode("utf-8", errors="replace")
            P(f"\n{'_'*60}")
            P(f"[{i+1}/{len(css_entries)}] {entry.path}")
            P(f"Group: {g} | Size: {len(data):,} bytes")
            P("_" * 60)
            P(text[:4000])
            if len(text) > 4000:
                P(f"... ({len(text)-4000:,} more chars)")
        except Exception as e:
            P(f"  ERROR: {e}")

    # ── PHASE 3: XML with UI/loc refs ──
    P(f"\n{'='*70}")
    P("  PHASE 3: XML files with UI/localization references")
    P("=" * 70)

    xml_entries = ui_files.get(".xml", [])
    P(f"Total XML: {len(xml_entries)}")

    xml_loc_count = 0
    for i, (g, entry) in enumerate(xml_entries):
        try:
            data = vfs.read_entry_data(entry)
            text = data.decode("utf-8", errors="replace")
            bn = os.path.basename(entry.path).lower()

            is_ui = any(kw in bn for kw in ("ui", "menu", "hud", "dialog",
                                             "widget", "loc", "string",
                                             "save", "load", "option",
                                             "setting", "tooltip", "popup"))
            has_loc = bool(re.search(r'(?:loc|string|text|label|strid)\w*\s*=', text, re.I))
            has_ui_tags = bool(re.search(r'<(?:Widget|Panel|Button|Window|Menu|Dialog|Text|Label)', text, re.I))

            if is_ui or has_loc or has_ui_tags:
                xml_loc_count += 1
                if xml_loc_count <= 200:  # Cap output
                    P(f"\n{'_'*60}")
                    P(f"[XML #{xml_loc_count}] {entry.path}")
                    P(f"Group: {g} | Size: {len(data):,} bytes")
                    P("_" * 60)
                    P(text[:3000])
                    if len(text) > 3000:
                        P(f"... ({len(text)-3000:,} more chars)")
        except:
            pass

    P(f"\nTotal XML with UI/loc refs: {xml_loc_count}")

    # ── PHASE 4: .uianiminit ──
    P(f"\n{'='*70}")
    P("  PHASE 4: .uianiminit files")
    P("=" * 70)

    uiani = ui_files.get(".uianiminit", [])
    P(f"Total: {len(uiani)}")

    for i, (g, entry) in enumerate(uiani[:50]):
        try:
            data = vfs.read_entry_data(entry)
            P(f"\n  [{i+1}] {entry.path} | {len(data):,} bytes")
            try:
                text = data.decode("utf-8")
                is_text = all(c.isprintable() or c in "\r\n\t" for c in text[:500])
            except:
                is_text = False
            if is_text:
                P(f"  TYPE: TEXT")
                P(f"  {text[:500]}")
            else:
                P(f"  TYPE: BINARY | Magic: {data[:8].hex()}")
                strings = re.findall(rb'[\x20-\x7e]{4,}', data[:4000])
                if strings:
                    P(f"  Strings: {[s.decode() for s in strings[:15]]}")
        except:
            pass

    # ── PHASE 5: .pami sample ──
    P(f"\n{'='*70}")
    P("  PHASE 5: .pami files (sample)")
    P("=" * 70)

    pami = ui_files.get(".pami", [])
    P(f"Total: {len(pami)}")

    for i, (g, entry) in enumerate(pami[:20]):
        try:
            data = vfs.read_entry_data(entry)
            text = data.decode("utf-8", errors="replace")
            P(f"\n{'_'*40}")
            P(f"[{i+1}] {entry.path} | {len(data):,} bytes")
            P(text[:1500])
        except:
            pass

    # ── PHASE 6: Paloc "save/load/menu" trace ──
    P(f"\n{'='*70}")
    P("  PHASE 6: Paloc entries with save/load/menu/setting/option")
    P("=" * 70)

    for g, entry in ui_files.get(".paloc", []):
        try:
            data = vfs.read_entry_data(entry)
            parsed = parse_paloc(data)
            hits = [e for e in parsed
                    if any(kw in e.value.lower() for kw in ("save", "load", "menu", "setting", "option"))
                    or any(kw in e.key.lower() for kw in ("save", "load", "menu", "setting", "option"))]

            if hits:
                P(f"\n  FILE: {entry.path} ({len(parsed)} total)")
                for h in hits[:100]:
                    P(f"    KEY: {h.key:50s} -> {h.value[:100]}")
        except:
            pass

    # ── PHASE 7: .txt files ──
    P(f"\n{'='*70}")
    P("  PHASE 7: .txt files")
    P("=" * 70)

    txt = ui_files.get(".txt", [])
    P(f"Total: {len(txt)}")
    for i, (g, entry) in enumerate(txt):
        try:
            data = vfs.read_entry_data(entry)
            text = data.decode("utf-8", errors="replace")
            P(f"\n  [{i+1}] {entry.path} | {len(data):,} bytes")
            P(f"  {text[:500]}")
        except:
            pass

    # ── Summary ──
    P(f"\n{'='*70}")
    P("  HTML/THTML DETECTED PATTERNS")
    P("=" * 70)
    for pattern, count in sorted(loc_patterns.items(), key=lambda x: -x[1])[:50]:
        P(f"  {pattern:50s} : {count:,}")

    f.close()
    print(f"Done! Output written to: {OUT_FILE}")


if __name__ == "__main__":
    run()
