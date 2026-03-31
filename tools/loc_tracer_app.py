"""Localization Tracer — type any text, see exactly where it lives in the game.

Enhanced: shows exact screen name, UI box, element, CSS styling, all locations,
deduped results, and full rendering chain.
"""

import os
import sys
import re
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QPushButton, QTextEdit, QSplitter, QTreeWidget,
    QTreeWidgetItem, QProgressBar, QGroupBox, QHeaderView, QComboBox,
    QStatusBar,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor

from core.vfs_manager import VfsManager
from core.paloc_parser import parse_paloc
from utils.platform_utils import auto_discover_game


# ═══════════════════════════════════════════════════════════════════════
#  GAME SCREEN NAME MAP — HTML file -> human-readable name + category
# ═══════════════════════════════════════════════════════════════════════

SCREEN_MAP = {
    # ── Main Menus ──
    "systemroot.html":          ("System Root",              "System",        "Root HTML container for the entire UI"),
    "titleview.html":           ("Title Screen",             "Main Menu",     "Game title/start screen"),
    "mainmenuview2.html":       ("Main Menu (ESC)",          "Main Menu",     "ESC menu with Inventory/Map/Skills/System buttons"),
    "mainmenuquickslotview.html": ("Main Menu Quick Slots",  "Main Menu",     "Quick access panel in ESC menu"),
    "mainmenudimmedview.html":  ("Main Menu Dimmed BG",      "Main Menu",     "Dimmed background overlay when ESC menu opens"),
    "logoutview.html":          ("Logout / Char Select",     "Main Menu",     "Logout and return to character select screen"),
    "gameexitpanel.html":       ("Exit Game Dialog",         "Main Menu",     "\"Are you sure you want to exit?\" confirmation"),
    "settingsview.html":        ("Settings",                 "Main Menu",     "Game settings (Graphics/Sound/Controls)"),
    "subsettingsview.html":     ("Sub-Settings",             "Main Menu",     "Settings sub-panels"),
    "inputcustomizingview.html":("Key Bindings",             "Main Menu",     "Custom key binding editor"),
    "keymapview.html":          ("Key Map View",             "Main Menu",     "Full keyboard layout display"),

    # ── Character ──
    "characterlistview.html":   ("Character Select",         "Character",     "Character list + save slot selection"),
    "gamedataloadview.html":    ("Load Game Data",           "Character",     "Save/Load game data screen"),
    "titlegamedataloadview.html": ("Title Load Game",        "Character",     "Load game from title screen"),

    # ── HUD / In-Game ──
    "hudview2.html":            ("HUD",                      "HUD",           "Main in-game HUD (health, mana, minimap)"),
    "hudoverpanel.html":        ("HUD Overlay",              "HUD",           "HUD overlay panel (quick slots, guides)"),
    "minimaphudview2.html":     ("Minimap",                  "HUD",           "Minimap in top-right corner"),
    "statusgaugeview2.html":    ("Status Gauges",            "HUD",           "HP/MP/Stamina gauge bars"),
    "crosshairautotargetingview.html": ("Crosshair",         "HUD",           "Auto-targeting crosshair overlay"),
    "combocountview.html":      ("Combo Counter",            "HUD",           "Hit combo counter display"),
    "warningindicatorview.html":("Warning Indicator",        "HUD",           "Danger/warning directional indicator"),
    "timerhud.html":            ("Timer HUD",                "HUD",           "In-game timer display"),
    "bossnametagview2.html":    ("Boss Name Tag",            "HUD",           "Boss enemy name + HP bar"),

    # ── Alerts / Notifications ──
    "alertsystemview.html":     ("Alert Popup",              "Alerts",        "Live notification popup (bottom of screen)"),
    "alerthistoryview.html":    ("Alert History",            "Alerts",        "Past notifications log (scrollable list)"),
    "alertsystemoverview.html": ("Alert System Over",        "Alerts",        "Alert overlay system"),
    "alertsystemunderview.html":("Alert System Under",       "Alerts",        "Alert under-layer system"),

    # ── Cinematics / Cutscenes ──
    "cinemaview.html":          ("Cinematic",                "Cinema",        "Cutscene letterbox with Skip/Stop buttons"),
    "gamepauseview.html":       ("Game Pause",               "Cinema",        "Pause overlay during gameplay"),
    "dialogsubtitleview.html":  ("Dialog Subtitles",         "Cinema",        "NPC dialogue subtitle bar"),
    "endingcreditsview.html":   ("Ending Credits",           "Cinema",        "End-game credits roll"),

    # ── Loading ──
    "gameloadingview.html":     ("Loading Screen",           "Loading",       "Game loading screen with tips"),
    "gamedataloadingview.html": ("Data Loading",             "Loading",       "Game data loading progress"),
    "gamestagefadescreen.html": ("Fade Screen",              "Loading",       "Black fade transition screen"),
    "transitiontitleview.html": ("Transition Title",         "Loading",       "Chapter/area transition title card"),
    "chaptertransitionpanel.html": ("Chapter Transition",    "Loading",       "Chapter change transition"),

    # ── Inventory / Items ──
    "inventoryequipmentpanel.html": ("Inventory",            "Inventory",     "Main inventory + equipment panel"),
    "itemquickslotview2.html":  ("Item Quick Slots",         "Inventory",     "Quick slot item selection (potions, etc.)"),
    "itemlogview2.html":        ("Item Log",                 "Inventory",     "Item acquisition log"),
    "itemenchantpanel.html":    ("Enchant",                  "Inventory",     "Equipment enchantment panel"),
    "itemsocketpanel.html":     ("Socket",                   "Inventory",     "Item socket gem panel"),
    "equipmentrepairview2.html":("Equipment Repair",         "Inventory",     "NPC equipment repair panel"),
    "dyeview.html":             ("Dye",                      "Inventory",     "Equipment dye/color panel"),
    "downloadcontentsview.html":("DLC Content",              "Inventory",     "DLC/premium content claim"),

    # ── Skills ──
    "skilltreepanel.html":      ("Skill Tree",               "Skills",        "Full skill tree panel"),
    "skilllistpanel.html":      ("Skill List",               "Skills",        "Skill list view"),

    # ── Quests ──
    "questmenupanel.html":      ("Quest Menu",               "Quests",        "Quest journal menu"),
    "questinventorymenuview.html": ("Quest Inventory",       "Quests",        "Quest-related items"),
    "dailyquestmenupanel.html": ("Daily Quests",             "Quests",        "Daily quest list"),
    "factionquestmenupanel.html": ("Faction Quests",         "Quests",        "Faction quest list"),
    "challengemenupanel.html":  ("Challenges",               "Quests",        "Challenge/achievement list"),
    "challengemenupanel2.html": ("Challenges v2",            "Quests",        "Challenge panel variant"),
    "journalview.html":         ("Journal",                  "Quests",        "Game journal/diary"),
    "extrarewardpanel.html":    ("Extra Rewards",            "Quests",        "Bonus reward panel"),

    # ── NPC / Shop ──
    "interactionpanel.html":    ("NPC Interaction",          "NPC",           "NPC interaction menu (talk/trade/etc.)"),
    "npcstorebuyview2.html":    ("NPC Shop (Buy)",           "NPC",           "Buy items from NPC"),
    "npcstoresellview2.html":   ("NPC Shop (Sell)",          "NPC",           "Sell items to NPC"),
    "npcstoretradebuyview.html":("NPC Trade Buy",            "NPC",           "NPC trade purchase"),
    "npcstoretradesellview.html": ("NPC Trade Sell",         "NPC",           "NPC trade sell"),
    "npcrestorationview.html":  ("NPC Restoration",          "NPC",           "NPC item restoration"),
    "bankpanel.html":           ("Bank / Storage",           "NPC",           "Bank/warehouse NPC panel"),
    "warehouseview2.html":      ("Warehouse",                "NPC",           "Item warehouse/storage"),
    "barbershopview.html":      ("Barber Shop",              "NPC",           "Character appearance editor"),
    "indulgenceview.html":      ("Indulgence Shop",          "NPC",           "Indulgence (karma reset) NPC"),
    "stablebuyview.html":       ("Stable Buy",               "NPC",           "Horse stable purchase"),
    "itemgiftinventory.html":   ("Gift Inventory",           "NPC",           "Gift-giving inventory panel"),
    "letterlistview.html" if False else
    "letterview.html":          ("Letters",                   "NPC",           "Letter/mail view"),

    # ── Mounts / Vehicles ──
    "horsestableview.html":     ("Horse Stable",             "Mount",         "Horse management + skills"),
    "vehicleequipmentview.html":("Vehicle Equipment",        "Mount",         "Mount equipment panel"),
    "vehicleinventoryview.html":("Vehicle Inventory",        "Mount",         "Mount inventory"),
    "vehiclemoveitempanel.html":("Vehicle Move Item",        "Mount",         "Move items to/from mount"),
    "petview.html":             ("Pet View",                 "Mount",         "Pet companion panel"),
    "pethouseview.html":        ("Pet House",                "Mount",         "Pet housing management"),

    # ── Faction ──
    "factionoperationmanagementpanel.html": ("Faction Operations", "Faction",  "Faction operation management"),
    "factionresearchpanel.html":("Faction Research",         "Faction",       "Faction research tree"),
    "factionreligionpanel.html":("Faction Religion",         "Faction",       "Religious faction panel"),
    "factiontradeagreementpanel.html": ("Trade Agreement",   "Faction",       "Faction trade deal panel"),
    "factiondonationpanel.html":("Faction Donation",         "Faction",       "Faction donation panel"),
    "farmmanagementpanel.html": ("Farm Management",          "Faction",       "Farm/agriculture panel"),
    "mercenarymanagementpanel.html": ("Mercenary Management","Faction",       "Mercenary hiring/management"),
    "workermanagementpanel.html": ("Worker Management",      "Faction",       "Worker assignment panel"),

    # ── World Map ──
    "worldmapview.html":        ("World Map",                "Map",           "Full world map"),
    "worldmapobserverpanel.html": ("World Map Observer",     "Map",           "World map observation panel"),
    "milepostview.html":        ("Milepost / Fast Travel",   "Map",           "Fast travel waypoint"),

    # ── Knowledge / Codex ──
    "knowledgepanel2.html":     ("Knowledge",                "Knowledge",     "Knowledge/codex panel"),
    "knowledgeview.html":       ("Knowledge View",           "Knowledge",     "Knowledge detail view"),
    "visionelibrarypanel.html": ("Visione Library",          "Knowledge",     "Memory fragment library"),
    "inspectview.html":         ("Inspect",                  "Knowledge",     "Object inspection panel"),

    # ── Crafting ──
    "cookingview3.html":        ("Cooking",                  "Crafting",      "Cooking crafting station"),
    "crafttransferview.html":   ("Craft Transfer",           "Crafting",      "Material transfer crafting"),
    "crafttransferview2.html":  ("Craft Transfer v2",        "Crafting",      "Material transfer variant"),
    "kukuenchantpanel.html":    ("Enchant (Kuku)",           "Crafting",      "Special enchantment panel"),
    "installmodeview.html":     ("Install Mode",             "Crafting",      "Housing/furniture placement"),
    "installselectpanel.html":  ("Install Select",           "Crafting",      "Object selection for placement"),
    "randomboxpanel.html":      ("Random Box",               "Crafting",      "Loot box opening"),

    # ── Combat / QTE ──
    "gameoverview2.html":       ("Game Over",                "Combat",        "Death/game over screen"),
    "changecharacternoticepanel.html": ("Change Character",  "Combat",        "Character switch notice"),
    "dialogpuzzle.html":        ("Dialog Puzzle",            "Combat",        "Dialogue puzzle/evidence"),
    "oneclickqteview.html":     ("QTE: One Click",           "Combat",        "Quick-time event (single press)"),
    "doubleclickqteview.html":  ("QTE: Double Click",        "Combat",        "Quick-time event (double press)"),
    "multiclickqteview.html":   ("QTE: Multi Click",         "Combat",        "Quick-time event (multi press)"),
    "pressclickqteview.html":   ("QTE: Press & Hold",        "Combat",        "Quick-time event (hold)"),
    "repeatclickqteview.html":  ("QTE: Repeat Click",        "Combat",        "Quick-time event (mash)"),
    "timingclickqteview.html":  ("QTE: Timing",              "Combat",        "Quick-time event (timed)"),
    "bartimingclickqteview.html": ("QTE: Bar Timing",        "Combat",        "Quick-time event (bar fill)"),
    "spinqteview.html":         ("QTE: Spin",                "Combat",        "Quick-time event (rotate)"),
    "balanceqteview.html":      ("QTE: Balance",             "Combat",        "Quick-time event (balance)"),
    "arrowshotpanel.html":      ("Arrow Shot Panel",         "Combat",        "Archery aiming panel"),

    # ── Mini-Games ──
    "cardminigameview.html":    ("Card Game",                "Mini-Game",     "Card mini-game"),
    "cardtexasholdempokerview.html": ("Texas Hold'em",       "Mini-Game",     "Poker mini-game"),
    "diceminigameview.html":    ("Dice Game",                "Mini-Game",     "Dice rolling mini-game"),
    "fishingminigameview.html": ("Fishing",                  "Mini-Game",     "Fishing mini-game"),
    "armwrestlingtimelimitview.html": ("Arm Wrestling",      "Mini-Game",     "Arm wrestling mini-game"),
    "duelminigameview.html":    ("Duel",                     "Mini-Game",     "Dueling mini-game"),
    "rockpapersciview.html":    ("Rock Paper Scissors",      "Mini-Game",     "RPS mini-game"),
    "slapfightview.html":       ("Slap Fight",               "Mini-Game",     "Slap fight mini-game"),
    "sixlominigameview.html":   ("Sixlo Game",               "Mini-Game",     "Sixlo board game"),
    "slotmachineminigameview.html": ("Slot Machine",         "Mini-Game",     "Slot machine mini-game"),
    "milkingcowview.html":      ("Milking Cow",              "Mini-Game",     "Cow milking mini-game"),
    "rodeominigameview.html":   ("Rodeo",                    "Mini-Game",     "Rodeo riding mini-game"),
    "jitgottaengminigameview.html": ("Jitgottaeng",          "Mini-Game",     "Korean traditional game"),
    "recitalview.html":         ("Music Recital",            "Mini-Game",     "Music performance"),
    "freerecitalpanel.html":    ("Free Recital",             "Mini-Game",     "Free music performance"),
    "sheetmusicselectionpanel.html": ("Sheet Music",         "Mini-Game",     "Sheet music selection"),

    # ── Guides / Tutorials ──
    "playguideview.html":       ("Play Guide",               "Guide",         "In-game control guide overlay"),
    "fullscreenguideview.html": ("Full Screen Guide",        "Guide",         "Full-screen tutorial/guide"),
    "tutorialview.html":        ("Tutorial",                 "Guide",         "Tutorial system"),
    "keyguidepanel.html":       ("Key Guide Panel",          "Guide",         "Button prompt guide"),
    "minigameguidepanel.html":  ("Mini-Game Guide",          "Guide",         "Mini-game instruction panel"),

    # ── Dialogs / Modals ──
    "modalmessageview.html":    ("Modal Message",            "Dialog",        "Generic confirmation/alert modal"),
    "buttonlistpanel.html":     ("Button List",              "Dialog",        "Multi-button selection list"),
    "countdownpanel.html":      ("Countdown",                "Dialog",        "Countdown timer panel"),
    "gaugepanel.html":          ("Gauge Panel",              "Dialog",        "Progress gauge panel"),
    "gimmickprogresspanel.html":("Gimmick Progress",         "Dialog",        "Puzzle/gimmick progress bar"),
    "clickeffectpanel.html":    ("Click Effect",             "Dialog",        "Click feedback effect panel"),
    "warningtimerpanel.html":   ("Warning Timer",            "Dialog",        "Timed warning panel"),
    "witnesswantedview.html":   ("Wanted Notice",            "Dialog",        "Criminal wanted/bounty notice"),
    "moneylogview.html":        ("Money Log",                "Dialog",        "Currency transaction log"),

    # ── Demo / Special ──
    "demoplaydescpanel.html":   ("Demo Description",         "Demo",          "Demo version description"),
    "demoplaymodalpanel.html":  ("Demo Modal",               "Demo",          "Demo version popup"),
    "demoplaytimerhud.html":    ("Demo Timer",               "Demo",          "Demo time remaining"),
    "watermarkview.html":       ("Watermark",                "System",        "Watermark overlay"),
    "licenseview.html":         ("License",                  "System",        "License/legal display"),
    "photomodepanel.html":      ("Photo Mode",               "System",        "Screenshot/photo mode"),

    # ── Templates (.thtml) ──
    "keyguide.thtml":           ("KeyGuide Template",        "Template",      "Reusable button prompt widget"),
    "modalmessage.thtml":       ("Modal Template",           "Template",      "Reusable modal dialog widget"),
    "itemtooltip2.thtml":       ("Item Tooltip Template",    "Template",      "Reusable item tooltip widget"),
    "itemlist.thtml":           ("Item List Template",        "Template",      "Reusable item list widget"),
    "itemicon.thtml":           ("Item Icon Template",        "Template",      "Reusable item icon widget"),
    "characterstat.thtml":      ("Character Stat Template",  "Template",      "Reusable character stat widget"),
    "npcinteractiontitle.thtml":("NPC Title Template",       "Template",      "Reusable NPC interaction header"),
    "cdcommon.thtml":           ("Common Template",          "Template",      "Shared common widgets"),
    "inventory2.thtml":         ("Inventory Template",       "Template",      "Reusable inventory widget"),
    "questinfo.thtml":          ("Quest Info Template",      "Template",      "Reusable quest info widget"),
    "selection.thtml":          ("Selection Template",       "Template",      "Reusable selection widget"),
    "tabbar.thtml":             ("Tab Bar Template",         "Template",      "Reusable tab bar widget"),
    "title.thtml":              ("Title Template",           "Template",      "Reusable title bar widget"),
    "gauge.thtml":              ("Gauge Template",           "Template",      "Reusable gauge bar widget"),
    "timer.thtml":              ("Timer Template",           "Template",      "Reusable timer widget"),
    "currency.thtml":           ("Currency Template",        "Template",      "Reusable currency display widget"),
    "pet.thtml":                ("Pet Template",             "Template",      "Reusable pet info widget"),
    "eula.thtml":               ("EULA Template",            "Template",      "End-user license agreement"),
    "circularlist.thtml":       ("Circular List Template",   "Template",      "Reusable circular list widget"),
    "quickslotitem.thtml":      ("Quick Slot Template",      "Template",      "Reusable quick slot item widget"),
    "builtincontroller.thtml":  ("Built-in Controller",      "Template",      "Base controller template"),
    "basecontrollereditor.thtml": ("Editor Controller",      "Template",      "Editor base controller"),
    "commontoy.thtml":          ("Common Toy Template",      "Template",      "Reusable toy/figurine widget"),
    "consolegamepad.thtml":     ("Console Gamepad",          "Template",      "Console controller template"),
    "system.thtml":             ("System Template",          "Template",      "System-level template"),
    "stageguideui.thtml":       ("Stage Guide Template",     "Template",      "Stage/level guide template"),
    "worldmapicon.thtml":       ("Map Icon Template",        "Template",      "Reusable map icon widget"),
    "minimapicon.thtml":        ("Minimap Icon Template",    "Template",      "Reusable minimap icon widget"),

    # ── Debug ──
    "commanddebugview.html":    ("Debug Commands",           "Debug",         "Developer QA test commands"),
    "devhudview.html":          ("Dev HUD",                  "Debug",         "Developer HUD overlay"),
    "devlogview.html":          ("Dev Log",                  "Debug",         "Developer log output"),
    "characterinfodebugview.html": ("Character Debug",       "Debug",         "Debug character info"),
    "debugsubtitlepanel.html":  ("Debug Subtitles",          "Debug",         "Debug subtitle overlay"),
}


def get_screen_info(html_path):
    """Get (screen_name, category, description) for an HTML file path."""
    basename = os.path.basename(html_path).lower()
    if basename in SCREEN_MAP:
        return SCREEN_MAP[basename]
    # Auto-generate from filename
    name = basename.replace(".html", "").replace(".thtml", "")
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = name.replace("view", "").replace("panel", "").replace("2", "").strip()
    return (name.title(), "Unknown", f"UI file: {basename}")


# ═══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HtmlLocalstring:
    localstring_key: str
    html_file: str
    element_tag: str
    element_id: str
    css_classes: list = field(default_factory=list)
    parent_context: str = ""
    full_line: str = ""
    group: str = ""
    override_selector: str = ""  # for <override> elements


@dataclass
class CssRule:
    selector: str
    properties: dict
    file: str
    group: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  INDEX BUILDER
# ═══════════════════════════════════════════════════════════════════════

class IndexBuilder(QThread):
    progress = Signal(int, str)
    finished = Signal(object)

    def __init__(self, packages_path):
        super().__init__()
        self._pkg = packages_path

    def run(self):
        index = {
            "paloc_entries": [],
            "paloc_by_value_lower": defaultdict(list),
            "paloc_by_key": {},
            "html_localstrings": [],
            "localstring_map": defaultdict(list),
            "css_rules": [],
            "css_by_selector": defaultdict(list),
            "html_raw": {},
            "css_raw": {},
        }

        self.progress.emit(5, "Loading VFS...")
        vfs = VfsManager(self._pkg)
        vfs.load_papgt()
        for g in vfs.list_package_groups():
            try:
                vfs.load_pamt(g)
            except:
                pass

        # ── Paloc ──
        self.progress.emit(20, "Indexing paloc...")
        eng_entry = None
        for g, pamt in vfs._pamt_cache.items():
            for entry in pamt.file_entries:
                if "localizationstring_eng" in entry.path.lower():
                    eng_entry = (g, entry)
                    break
            if eng_entry:
                break

        if eng_entry:
            g, entry = eng_entry
            data = vfs.read_entry_data(entry)
            parsed = parse_paloc(data)
            for pe in parsed:
                rec = (pe.key, pe.value, entry.path)
                index["paloc_entries"].append(rec)
                index["paloc_by_key"][pe.key] = rec
                index["paloc_by_value_lower"][pe.value.lower().strip()].append(rec)

        # ── HTML/THTML ──
        self.progress.emit(40, "Indexing HTML/THTML...")
        html_entries = []
        for g, pamt in vfs._pamt_cache.items():
            for entry in pamt.file_entries:
                ext = os.path.splitext(entry.path.lower())[1]
                if ext in (".html", ".thtml"):
                    html_entries.append((g, entry))

        seen_ls = set()  # deduplicate

        for i, (g, entry) in enumerate(html_entries):
            try:
                data = vfs.read_entry_data(entry)
                text = data.decode("utf-8", errors="replace")
                index["html_raw"][entry.path] = text

                # Extract <tag ... localstring="KEY" ...>
                for m in re.finditer(r'<(\w+)\s+([^>]*?)localstring="([^"]*)"([^>]*?)/?>', text, re.DOTALL):
                    tag = m.group(1)
                    all_attrs = m.group(2) + m.group(4)
                    ls_key = m.group(3)

                    id_m = re.search(r'id="([^"]*)"', all_attrs)
                    elem_id = id_m.group(1) if id_m else ""
                    cls_m = re.search(r'class="([^"]*)"', all_attrs)
                    classes = cls_m.group(1).split() if cls_m else []
                    sel_m = re.search(r'selector="([^"]*)"', all_attrs)
                    override_sel = sel_m.group(1) if sel_m else ""

                    # Dedup key: (localstring, file, element_id or override_selector)
                    dedup = (ls_key, entry.path, elem_id or override_sel)
                    if dedup in seen_ls:
                        continue
                    seen_ls.add(dedup)

                    start = max(0, m.start() - 200)
                    end = min(len(text), m.end() + 200)

                    ls = HtmlLocalstring(
                        localstring_key=ls_key,
                        html_file=entry.path,
                        element_tag=tag,
                        element_id=elem_id,
                        css_classes=classes,
                        parent_context=text[start:end],
                        full_line=m.group(0),
                        group=g,
                        override_selector=override_sel,
                    )
                    index["html_localstrings"].append(ls)
                    index["localstring_map"][ls_key].append(ls)
            except:
                pass

        # ── CSS ──
        self.progress.emit(70, "Indexing CSS...")
        for g, pamt in vfs._pamt_cache.items():
            for entry in pamt.file_entries:
                if entry.path.lower().endswith(".css"):
                    try:
                        data = vfs.read_entry_data(entry)
                        text = data.decode("utf-8", errors="replace")
                        index["css_raw"][entry.path] = text
                        clean = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
                        for m in re.finditer(r'([^{}]+)\{([^}]*)\}', clean):
                            sel_raw = m.group(1).strip()
                            props = {}
                            for pm in re.finditer(r'([\w-]+)\s*:\s*([^;]+)', m.group(2)):
                                props[pm.group(1).strip()] = pm.group(2).strip()
                            for sel in sel_raw.split(","):
                                sel = sel.strip()
                                if sel and not sel.startswith("@"):
                                    rule = CssRule(sel, props, entry.path, g)
                                    index["css_rules"].append(rule)
                                    index["css_by_selector"][sel].append(rule)
                    except:
                        pass

        self.progress.emit(100, "Ready!")
        self.finished.emit(index)


# ═══════════════════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════════════════

THEME = """
QMainWindow, QWidget {
    background-color: #1e1e2e; color: #cdd6f4;
    font-family: "Segoe UI", "Consolas", monospace; font-size: 13px;
}
QLineEdit {
    background-color: #313244; color: #cdd6f4;
    border: 2px solid #45475a; border-radius: 8px;
    padding: 10px 14px; font-size: 16px;
    selection-background-color: #89b4fa;
}
QLineEdit:focus { border-color: #89b4fa; }
QTreeWidget {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #313244; border-radius: 6px;
    alternate-background-color: #1e1e2e; font-size: 12px;
}
QTreeWidget::item { padding: 4px 8px; }
QTreeWidget::item:selected { background-color: #313244; color: #89b4fa; }
QTreeWidget::item:hover { background-color: #2a2a3e; }
QHeaderView::section {
    background-color: #181825; color: #a6adc8;
    padding: 6px 10px; border: none; border-bottom: 1px solid #313244;
    font-weight: 600; font-size: 11px;
}
QTextEdit {
    background-color: #181825; color: #cdd6f4;
    border: 1px solid #313244; border-radius: 6px; padding: 8px;
    font-family: "Consolas", "Cascadia Code", monospace; font-size: 12px;
}
QProgressBar {
    background-color: #313244; border: none; border-radius: 4px;
    height: 6px; color: transparent;
}
QProgressBar::chunk { background-color: #89b4fa; border-radius: 4px; }
QLabel#title { font-size: 20px; font-weight: 700; color: #89b4fa; }
QLabel#subtitle { font-size: 12px; color: #6c7086; }
QLabel#section { font-size: 13px; font-weight: 600; color: #f9e2af; padding-top: 6px; }
QPushButton {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px; padding: 6px 16px;
}
QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
QComboBox {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px; padding: 4px 10px;
}
QStatusBar {
    background-color: #181825; color: #6c7086; border-top: 1px solid #313244;
}
"""


# ═══════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════

class LocTracerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CrimsonForge Localization Tracer")
        self.setMinimumSize(1200, 800)
        self.resize(1500, 950)
        self.setStyleSheet(THEME)
        self._index = None
        self._setup_ui()
        self._start_indexing()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 8)
        layout.setSpacing(10)

        title = QLabel("Localization Tracer")
        title.setObjectName("title")
        layout.addWidget(title)

        sub = QLabel("Type any text \u2192 see every place it appears in-game: screen, UI box, element, CSS style")
        sub.setObjectName("subtitle")
        layout.addWidget(sub)

        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Type any text... (Save, Load, Options, Confirm, attack, mount)")
        self._search.returnPressed.connect(self._do_search)
        self._search.textChanged.connect(self._on_text_changed)
        search_row.addWidget(self._search, 1)

        self._search_mode = QComboBox()
        self._search_mode.addItems(["Search Values (text shown in-game)",
                                     "Search Keys (paloc key IDs)",
                                     "Search Localstring Names (UI_ names)"])
        search_row.addWidget(self._search_mode)
        layout.addLayout(search_row)

        self._progress = QProgressBar()
        self._progress.setMaximum(100)
        layout.addWidget(self._progress)
        self._progress_label = QLabel("Loading game data...")
        self._progress_label.setObjectName("subtitle")
        layout.addWidget(self._progress_label)

        splitter = QSplitter(Qt.Horizontal)

        # Left: Results
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._result_count = QLabel("")
        self._result_count.setObjectName("section")
        left_layout.addWidget(self._result_count)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Game Screen / Location", "Element", "Localstring / Key", "CSS Style"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setColumnCount(4)
        h = self._tree.header()
        h.setStretchLastSection(True)
        h.resizeSection(0, 280)
        h.resizeSection(1, 200)
        h.resizeSection(2, 250)
        self._tree.currentItemChanged.connect(self._on_item_selected)
        left_layout.addWidget(self._tree)
        splitter.addWidget(left)

        # Right: Detail
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._detail_label = QLabel("Full Context")
        self._detail_label.setObjectName("section")
        right_layout.addWidget(self._detail_label)
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        right_layout.addWidget(self._detail)
        splitter.addWidget(right)

        splitter.setSizes([750, 500])
        layout.addWidget(splitter, 1)
        self._status = QStatusBar()
        self.setStatusBar(self._status)

    def _start_indexing(self):
        pkg = auto_discover_game()
        if not pkg:
            self._progress_label.setText("ERROR: Game not found!")
            self._progress_label.setStyleSheet("color: #f38ba8;")
            return
        self._progress_label.setText(f"Indexing: {pkg}")
        self._builder = IndexBuilder(pkg)
        self._builder.progress.connect(lambda p, m: (self._progress.setValue(p), self._progress_label.setText(m)))
        self._builder.finished.connect(self._on_index_done)
        self._builder.start()

    def _on_index_done(self, index):
        self._index = index
        self._progress.setVisible(False)
        n_paloc = len(index["paloc_entries"])
        n_ls = len(index["html_localstrings"])
        n_css = len(index["css_rules"])
        n_html = len(index["html_raw"])
        self._progress_label.setText(
            f"Ready | {n_paloc:,} strings | {n_ls} UI bindings | "
            f"{n_css:,} CSS rules | {n_html} HTML files"
        )
        self._progress_label.setStyleSheet("color: #a6e3a1;")
        self._search.setFocus()

    def _on_text_changed(self, text):
        if self._index and len(text) >= 2:
            if hasattr(self, '_timer'):
                self._timer.stop()
            self._timer = QTimer()
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._do_search)
            self._timer.start(400)

    def _do_search(self):
        if not self._index:
            return
        q = self._search.text().strip()
        if len(q) < 2:
            return
        self._tree.clear()
        self._detail.clear()

        ql = q.lower()
        mode = self._search_mode.currentIndex()

        # ── Find paloc hits ──
        paloc_hits = []
        if mode == 0:  # values
            for key, val, f in self._index["paloc_entries"]:
                if ql in val.lower():
                    paloc_hits.append((key, val, f))
        elif mode == 1:  # keys
            for key, val, f in self._index["paloc_entries"]:
                if ql in key.lower():
                    paloc_hits.append((key, val, f))
        elif mode == 2:  # localstring names
            for ls_key, ls_list in self._index["localstring_map"].items():
                if ql in ls_key.lower():
                    for ls in ls_list:
                        paloc_hits.append((ls_key, f"[localstring in {os.path.basename(ls.html_file)}]", ""))

        # ── Find HTML hits (exact localstring key matches from paloc values) ──
        html_hits = []
        if mode == 0:
            # For exact value matches, find localstring keys that could map
            for ls_key, ls_list in self._index["localstring_map"].items():
                if ql in ls_key.lower():
                    html_hits.extend(ls_list)
            # Also search raw HTML content
            for path, text in self._index["html_raw"].items():
                if ql in text.lower():
                    for m in re.finditer(re.escape(q), text, re.IGNORECASE):
                        line_start = text.rfind("\n", 0, m.start()) + 1
                        line_end = text.find("\n", m.end())
                        if line_end < 0:
                            line_end = len(text)
                        line = text[line_start:line_end].strip()
                        ls = HtmlLocalstring(
                            localstring_key="",
                            html_file=path,
                            element_tag="content",
                            element_id="",
                            css_classes=[],
                            parent_context=line,
                            full_line=line[:200],
                        )
                        html_hits.append(ls)
                        if len(html_hits) > 300:
                            break
        elif mode == 2:
            for ls_key, ls_list in self._index["localstring_map"].items():
                if ql in ls_key.lower():
                    html_hits.extend(ls_list)

        # ── Build tree ──
        total = len(paloc_hits) + len(html_hits)
        self._result_count.setText(f"Results: {total} hits")

        # === PALOC RESULTS ===
        if paloc_hits:
            paloc_root = QTreeWidgetItem(self._tree)
            paloc_root.setText(0, f"PALOC ENTRIES ({len(paloc_hits)})")
            paloc_root.setForeground(0, QColor("#a6e3a1"))
            paloc_root.setExpanded(True)

            # Group by exact value for clarity
            by_value = defaultdict(list)
            for key, val, f in paloc_hits[:200]:
                by_value[val].append((key, f))

            for val, keys in sorted(by_value.items(), key=lambda x: x[0].lower()):
                if len(keys) == 1:
                    key, f = keys[0]
                    item = QTreeWidgetItem(paloc_root)
                    item.setText(0, f'"{val[:60]}"')
                    item.setText(1, "Numeric ID" if key.isdigit() else "Symbolic Key")
                    item.setText(2, str(key))
                    item.setText(3, os.path.basename(f) if f else "")
                    item.setForeground(0, QColor("#cdd6f4"))
                    item.setForeground(2, QColor("#f9e2af") if key.isdigit() else QColor("#89b4fa"))
                    item.setData(0, Qt.UserRole, {"type": "paloc", "key": key, "value": val, "file": f})
                else:
                    val_item = QTreeWidgetItem(paloc_root)
                    val_item.setText(0, f'"{val[:60]}"')
                    val_item.setText(1, f"{len(keys)} entries")
                    val_item.setForeground(0, QColor("#cdd6f4"))
                    for key, f in keys:
                        child = QTreeWidgetItem(val_item)
                        child.setText(0, "")
                        child.setText(1, "Numeric ID" if key.isdigit() else "Symbolic Key")
                        child.setText(2, str(key))
                        child.setText(3, os.path.basename(f) if f else "")
                        child.setForeground(2, QColor("#f9e2af") if key.isdigit() else QColor("#89b4fa"))
                        child.setData(0, Qt.UserRole, {"type": "paloc", "key": key, "value": val, "file": f})

        # === HTML RESULTS (grouped by screen) ===
        if html_hits:
            html_root = QTreeWidgetItem(self._tree)
            html_root.setText(0, f"IN-GAME LOCATIONS ({len(html_hits)})")
            html_root.setForeground(0, QColor("#89b4fa"))
            html_root.setExpanded(True)

            # Group by screen (file)
            by_screen = defaultdict(list)
            for ls in html_hits:
                by_screen[ls.html_file].append(ls)

            # Sort by category then screen name
            def screen_sort_key(path):
                sname, cat, _ = get_screen_info(path)
                return (cat, sname)

            for fpath in sorted(by_screen.keys(), key=screen_sort_key):
                ls_list = by_screen[fpath]
                sname, category, description = get_screen_info(fpath)

                screen_item = QTreeWidgetItem(html_root)
                screen_item.setText(0, f"[{category}] {sname}")
                screen_item.setText(1, os.path.basename(fpath))
                screen_item.setText(2, f"{len(ls_list)} element(s)")
                screen_item.setText(3, description)
                screen_item.setForeground(0, QColor("#cba6f7"))
                screen_item.setExpanded(len(ls_list) <= 8)

                for ls in ls_list:
                    child = QTreeWidgetItem(screen_item)

                    # Element description
                    if ls.element_tag == "override":
                        elem_desc = f"<override> \u2192 {ls.override_selector or ls.element_id}"
                    elif ls.element_tag == "widget":
                        elem_desc = f"<widget> id={ls.element_id}"
                    elif ls.element_tag == "content":
                        elem_desc = "Raw content match"
                    else:
                        elem_desc = f"<{ls.element_tag}>"
                        if ls.element_id:
                            elem_desc += f" id={ls.element_id}"

                    # CSS summary
                    css_summary = ""
                    for cls in ls.css_classes:
                        if cls.startswith("font-"):
                            # Extract key visual info
                            sel = f".{cls}"
                            rules = self._index["css_by_selector"].get(sel, [])
                            if rules:
                                r = rules[0]
                                parts = []
                                if "font-size" in r.properties:
                                    parts.append(r.properties["font-size"])
                                if "color" in r.properties:
                                    parts.append(r.properties["color"])
                                if parts:
                                    css_summary = f"{cls} ({', '.join(parts)})"
                                    break
                    if not css_summary and ls.css_classes:
                        css_summary = " ".join(ls.css_classes[:3])

                    child.setText(0, elem_desc)
                    child.setText(1, ls.localstring_key if ls.localstring_key else ls.full_line[:60])
                    child.setText(2, ls.localstring_key if ls.localstring_key else "")
                    child.setText(3, css_summary)

                    if ls.localstring_key:
                        child.setForeground(1, QColor("#f9e2af"))

                    child.setData(0, Qt.UserRole, {"type": "html", "ls": ls})

        self._status.showMessage(f"Found {total} results for '{q}'")

    def _on_item_selected(self, current, _prev):
        if not current:
            return
        data = current.data(0, Qt.UserRole)
        if not data:
            return

        lines = []

        if data["type"] == "paloc":
            key, val, f = data["key"], data["value"], data["file"]
            lines.append("=" * 65)
            lines.append("  PALOC ENTRY — Localized Text")
            lines.append("=" * 65)
            lines.append(f"  Key:    {key}")
            lines.append(f"  Value:  {val}")
            lines.append(f"  File:   {f}")
            lines.append(f"  Type:   {'Numeric (game data ID)' if key.isdigit() else 'Symbolic (dialogue/quest)'}")
            lines.append("")
            lines.append("  In-game, the engine replaces template vars:")
            lines.append("    {{Key:Key_xxx}}     -> actual keybind (e.g. T, Left Click)")
            lines.append("    {{Money:xxx:N}}     -> currency icon + amount")
            lines.append("    {{Staticinfo:...}}  -> clickable game term link")
            lines.append("    <br/>               -> line break")

            # Find ALL screens where this text appears
            lines.append("")
            lines.append("=" * 65)
            lines.append("  WHERE THIS TEXT APPEARS IN-GAME")
            lines.append("=" * 65)

            # Search localstring map for keys that contain words from the value
            words = set(w.lower() for w in re.findall(r'[A-Za-z]{4,}', val))
            found_screens = []

            for ls_key, ls_list in self._index["localstring_map"].items():
                key_words = set(w.lower() for w in re.findall(r'[A-Za-z]{3,}', ls_key))
                if words & key_words:
                    for ls in ls_list:
                        found_screens.append((ls_key, ls))

            if found_screens:
                seen = set()
                for ls_key, ls in found_screens:
                    sname, cat, desc = get_screen_info(ls.html_file)
                    dedup = (sname, ls_key)
                    if dedup in seen:
                        continue
                    seen.add(dedup)

                    lines.append(f"\n  [{cat}] {sname}")
                    lines.append(f"  File:        {ls.html_file}")
                    lines.append(f"  Localstring: {ls_key}")
                    if ls.element_tag == "override":
                        lines.append(f"  Widget:      <override> targeting {ls.override_selector or ls.element_id}")
                    else:
                        lines.append(f"  Element:     <{ls.element_tag}> id=\"{ls.element_id}\"")
                    if ls.css_classes:
                        lines.append(f"  CSS Classes: {' '.join(ls.css_classes)}")
                        # Resolve CSS
                        for cls in ls.css_classes:
                            sel = f".{cls}"
                            rules = self._index["css_by_selector"].get(sel, [])
                            if rules:
                                r = rules[0]
                                props = "; ".join(f"{k}: {v}" for k, v in r.properties.items())
                                lines.append(f"    .{cls} {{ {props} }}")
                    lines.append(f"  Raw HTML:    {ls.full_line[:150]}")
            else:
                lines.append("")
                lines.append("  No direct localstring mapping found in HTML.")
                lines.append("  This value is loaded by the C++ engine at runtime.")
                lines.append("  Changing this paloc value will update the in-game text.")

        elif data["type"] == "html":
            ls = data["ls"]
            sname, cat, desc = get_screen_info(ls.html_file)

            lines.append("=" * 65)
            lines.append(f"  IN-GAME LOCATION")
            lines.append("=" * 65)
            lines.append(f"  Screen:       [{cat}] {sname}")
            lines.append(f"  Description:  {desc}")
            lines.append(f"  File:         {ls.html_file}")
            lines.append(f"  Group:        {ls.group}")
            lines.append("")

            lines.append("- " * 33)
            lines.append("  ELEMENT")
            lines.append("- " * 33)
            if ls.element_tag == "override":
                lines.append(f"  Type:         <override> (injects text into template widget)")
                lines.append(f"  Target:       {ls.override_selector or ls.element_id}")
                lines.append(f"  This replaces the text inside the widget's target element.")
            elif ls.element_tag == "widget":
                lines.append(f"  Type:         <widget> (reusable UI component)")
                lines.append(f"  Widget ID:    {ls.element_id}")
            elif ls.element_tag == "content":
                lines.append(f"  Type:         Raw text content match")
            else:
                lines.append(f"  Type:         <{ls.element_tag}>")
                lines.append(f"  ID:           {ls.element_id}")

            if ls.localstring_key:
                lines.append(f"  Localstring:  {ls.localstring_key}")
                lines.append(f"  The engine hashes this name to a numeric paloc key at runtime.")
            lines.append(f"\n  Raw HTML:\n    {ls.full_line}")

            # CSS
            if ls.css_classes:
                lines.append("")
                lines.append("- " * 33)
                lines.append("  CSS STYLING")
                lines.append("- " * 33)
                for cls in ls.css_classes:
                    if cls.startswith("cpp-"):
                        lines.append(f"\n  .{cls}  (C++ engine hook — no visual CSS)")
                        continue
                    sel = f".{cls}"
                    rules = self._index["css_by_selector"].get(sel, [])
                    if rules:
                        # Show one rule, note which file
                        r = rules[0]
                        lines.append(f"\n  .{cls} {{")
                        for k, v in r.properties.items():
                            lines.append(f"    {k}: {v};")
                        lines.append(f"  }}")
                        lines.append(f"  Source: {os.path.basename(r.file)}")
                        if len(rules) > 1:
                            other_files = set(os.path.basename(x.file) for x in rules[1:])
                            lines.append(f"  Also in: {', '.join(other_files)}")
                    else:
                        lines.append(f"\n  .{cls}  (no CSS rule found — may be in template)")

                # #id selector
                if ls.element_id and not ls.element_id.startswith("#"):
                    sel = f"#{ls.element_id}"
                    rules = self._index["css_by_selector"].get(sel, [])
                    if rules:
                        r = rules[0]
                        lines.append(f"\n  #{ls.element_id} {{")
                        for k, v in r.properties.items():
                            lines.append(f"    {k}: {v};")
                        lines.append(f"  }}")
                        lines.append(f"  Source: {os.path.basename(r.file)}")

            # Other screens with same localstring
            if ls.localstring_key and ls.localstring_key in self._index["localstring_map"]:
                all_locs = self._index["localstring_map"][ls.localstring_key]
                if len(all_locs) > 1:
                    lines.append("")
                    lines.append("- " * 33)
                    lines.append(f"  ALL SCREENS USING \"{ls.localstring_key}\"")
                    lines.append("- " * 33)
                    for other in all_locs:
                        sn, ct, _ = get_screen_info(other.html_file)
                        marker = " <-- YOU ARE HERE" if other.html_file == ls.html_file and other.element_id == ls.element_id else ""
                        lines.append(f"  [{ct}] {sn} ({os.path.basename(other.html_file)}){marker}")

            # Context
            if ls.parent_context:
                lines.append("")
                lines.append("- " * 33)
                lines.append("  SURROUNDING HTML CONTEXT")
                lines.append("- " * 33)
                lines.append(ls.parent_context)

        self._detail.setPlainText("\n".join(lines))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LocTracerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
