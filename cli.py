# cli.py - Command Line Interface for running with presets

import sys
import argparse
import json
import os
from typing import Optional


def load_presets():
    """Load presets from file"""
    presets_file = os.path.join(os.path.dirname(__file__), "llm_presets.json")

    if os.path.exists(presets_file):
        try:
            with open(presets_file, 'r', encoding='utf-8') as f: # Added encoding
                return json.load(f)
        except (IOError, OSError) as e: # Catch file-related errors
            print(f"Error: Could not read presets file '{presets_file}': {e}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in presets file '{presets_file}': {e}", file=sys.stderr)
        except Exception as e: # Catch any other unexpected error
            print(f"Error: Unexpected error loading presets from '{presets_file}': {e}", file=sys.stderr)
    else:
        # It's not an error if the file doesn't exist, just means no presets saved yet.
        pass
    return []


def list_presets():
    """List available presets"""
    try:
        presets = load_presets()
    except Exception as e: # Should be caught by load_presets, but as a defensive measure
        print(f"Critical error: Failed to load presets for listing: {e}", file=sys.stderr)
        return

    if not presets:
        print("No presets found. You can create presets by running the main application or using the --create flag.")
        return

    print("\nAvailable Presets:")
    print("-" * 50)

    for i, preset in enumerate(presets, 1):
        provider = preset.get('provider', 'unknown')
        model = preset.get('model', 'default')

        print(f"{i}. {preset['name']}")
        print(f"   Provider: {provider}")
        if model:
            print(f"   Model: {model}")
        print()


def run_with_preset(preset_name: str, skip_market_data: bool = False, demo_mode: bool = False):
    """Run application with specific preset"""
    try:
        presets = load_presets()
    except Exception as e: # Defensive
        print(f"Critical error: Failed to load presets before running: {e}", file=sys.stderr)
        sys.exit(1)

    # Find preset
    preset = None
    for p in presets:
        if p['name'].lower() == preset_name.lower():
            preset = p
            break

    if not preset:
        print(f"Error: Preset '{preset_name}' not found.")
        print("Use --list to see available presets.")
        sys.exit(1)

    # Create startup config
    config = {
        "provider": preset.get('provider'),
        "api_key": preset.get('api_key', ''),
        "model": preset.get('model', ''),
        "settings": preset.get('settings', {}),
        "skip_market_data": skip_market_data,
        "demo_mode": demo_mode,
        "use_preset": True
    }

    try:
        # Launch application with config
        from PyQt6.QtWidgets import QApplication
        from main_window import MainWindow # Assuming main_window.py handles its own startup errors

        app = QApplication(sys.argv)
        app.setApplicationName("AI Stock Analyzer")
        app.setOrganizationName("StockAnalyzer") # Used by QSettings
        app.setStyle("Fusion") # Optional: set a consistent style

        window = MainWindow(startup_config=config)
        window.show()
        sys.exit(app.exec())
    except ImportError as e:
        print(f"Error: Failed to import PyQt6 or other GUI components: {e}. "
              "Please ensure they are installed correctly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while trying to launch the application with preset '{preset_name}': {e}", file=sys.stderr)
        # import traceback # For debugging, uncomment to see full traceback
        # traceback.print_exc()
        sys.exit(1)


def create_preset_file(name: str, provider: str, api_key: str, model: Optional[str] = None):
    """Create a new preset from command line"""
    try:
        presets = load_presets()
    except Exception as e: # Defensive
        print(f"Warning: Failed to load existing presets: {e}. A new presets file might be created.", file=sys.stderr)
        presets = []

    # Check if preset already exists
    for p in presets:
        if p['name'].lower() == name.lower():
            print(f"Error: Preset '{name}' already exists.")
            return

    # Create new preset
    preset = {
        "name": name,
        "provider": provider,
        "api_key": api_key
    }

    if model:
        preset["model"] = model

    if provider == "lmstudio":
        preset["settings"] = {"url": api_key}  # For LM Studio, api_key is actually the URL
        preset["api_key"] = ""

    presets.append(preset)

    presets_file = os.path.join(os.path.dirname(__file__), "llm_presets.json")
    try:
        with open(presets_file, 'w', encoding='utf-8') as f: # Added encoding
            json.dump(presets, f, indent=2)
        print(f"✅ Preset '{name}' created successfully!")
    except (IOError, OSError) as e:
        print(f"Error: Could not write presets file '{presets_file}': {e}", file=sys.stderr)
    except Exception as e: # Catch any other unexpected error during save
        print(f"Error: Unexpected error saving preset '{name}': {e}", file=sys.stderr)


def main():
    # No top-level logging basicConfig here; let main_window or specific modules handle if GUI runs.
    # CLI output should primarily be to stdout/stderr.

    parser = argparse.ArgumentParser(
        description="AI Stock Analyzer - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all presets
  python cli.py --list
  
  # Run with a specific preset
  python cli.py --preset "Quick Start - OpenAI"
  
  # Run with preset in demo mode
  python cli.py --preset "Local - LM Studio" --demo
  
  # Create a new preset
  python cli.py --create "My GPT-4" --provider openai --api-key "sk-..." --model gpt-4
  
  # Create a local LM Studio preset
  python cli.py --create "My Local LLM" --provider lmstudio --api-key "http://localhost:1234"
        """
    )

    parser.add_argument('--list', action='store_true', help='List available presets')
    parser.add_argument('--preset', type=str, help='Run with specific preset name')
    parser.add_argument('--skip-market-data', action='store_true', help='Skip initial market data download')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with mock data')

    # Preset creation arguments
    parser.add_argument('--create', type=str, help='Create a new preset with given name')
    parser.add_argument('--provider', type=str, choices=['openai', 'anthropic', 'gemini', 'huggingface', 'lmstudio'])
    parser.add_argument('--api-key', type=str, help='API key for the provider (or URL for LM Studio)')
    parser.add_argument('--model', type=str, help='Model name (optional)')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f"Argument parsing error: {e}", file=sys.stderr)
        sys.exit(2)
    except SystemExit as e:
        sys.exit(e.code)


    try:
        if args.list:
            list_presets()
        elif args.preset:
            run_with_preset(args.preset, args.skip_market_data, args.demo)
        elif args.create:
            if not args.provider or not args.api_key:
                parser.error("--provider and --api-key are required when using --create.")
            create_preset_file(args.create, args.provider, args.api_key, args.model)
        else:
            from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
            from startup_config import StartupConfigDialog
            from main_window import MainWindow

            app = QApplication(sys.argv)
            app.setApplicationName("AI Stock Analyzer")
            app.setOrganizationName("StockAnalyzer")
            app.setStyle("Fusion")

            config_dialog = StartupConfigDialog()
            config = None

            def on_config_complete(cfg):
                nonlocal config
                config = cfg

            config_dialog.config_complete.connect(on_config_complete)

            if config_dialog.exec() != QDialog.DialogCode.Accepted:
                print("Startup configuration cancelled by user. Exiting.", file=sys.stderr)
                sys.exit(0)

            if not config:
                print("Error: Configuration was not set after dialog. Cannot start application.", file=sys.stderr)
                sys.exit(1)

            if args.skip_market_data:
                config['skip_market_data'] = True
            if args.demo:
                config['demo_mode'] = True

            window = MainWindow(startup_config=config)
            window.show()
            sys.exit(app.exec())

    except ImportError as e:
        print(f"Error: A required module (likely PyQt6 or a dependency) is missing: {e}. "
              "Please ensure all dependencies are installed.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e}. Please check your installation.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse a critical JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred at the CLI top level: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
