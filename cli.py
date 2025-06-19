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
            with open(presets_file, 'r') as f:
                return json.load(f)
        except:
            pass

    return []


def list_presets():
    """List available presets"""
    presets = load_presets()

    if not presets:
        print("No presets found. Run the application to create presets.")
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
    presets = load_presets()

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

    # Launch application with config
    from PyQt6.QtWidgets import QApplication
    from main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AI Stock Analyzer")
    app.setOrganizationName("StockAnalyzer")
    app.setStyle("Fusion")

    # Create main window with config
    window = MainWindow(startup_config=config)
    window.show()

    sys.exit(app.exec())


def create_preset_file(name: str, provider: str, api_key: str, model: Optional[str] = None):
    """Create a new preset from command line"""
    presets = load_presets()

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

    # Save presets
    presets_file = os.path.join(os.path.dirname(__file__), "llm_presets.json")
    try:
        with open(presets_file, 'w') as f:
            json.dump(presets, f, indent=2)
        print(f"✅ Preset '{name}' created successfully!")
    except Exception as e:
        print(f"Error saving preset: {e}")


def main():
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

    args = parser.parse_args()

    if args.list:
        list_presets()
    elif args.preset:
        run_with_preset(args.preset, args.skip_market_data, args.demo)
    elif args.create:
        if not args.provider or not args.api_key:
            print("Error: --provider and --api-key are required when creating a preset")
            sys.exit(1)
        create_preset_file(args.create, args.provider, args.api_key, args.model)
    else:
        # Run normal startup flow
        from PyQt6.QtWidgets import QApplication, QDialog
        from startup_config import StartupConfigDialog
        from main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("AI Stock Analyzer")
        app.setOrganizationName("StockAnalyzer")
        app.setStyle("Fusion")

        # Show startup configuration dialog
        config_dialog = StartupConfigDialog()

        config = None

        def on_config_complete(cfg):
            nonlocal config
            config = cfg

        config_dialog.config_complete.connect(on_config_complete)

        if config_dialog.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)

        if not config:
            print("Error: No configuration provided")
            sys.exit(1)

        # Add command line flags to config
        if args.skip_market_data:
            config['skip_market_data'] = True
        if args.demo:
            config['demo_mode'] = True

        # Create and show main window
        window = MainWindow(startup_config=config)
        window.show()

        sys.exit(app.exec())


if __name__ == "__main__":
    main()
