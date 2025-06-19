# startup_config.py - Startup Configuration Dialog

import json
import os
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QGroupBox, QFormLayout, QTextEdit,
    QDialogButtonBox, QMessageBox, QListWidget, QListWidgetItem,
    QStackedWidget, QWidget, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

class LLMPreset:
    """LLM configuration preset"""
    def __init__(self, name: str, provider: str, model: str = "", api_key: str = "", settings: Optional[Dict[str, Any]] = None): # Allow None for settings
        try:
            if not name or not isinstance(name, str):
                raise ValueError("Preset name must be a non-empty string.")
            if not provider or not isinstance(provider, str):
                raise ValueError("Provider must be a non-empty string.")

            self.name = name
            self.provider = provider
            self.model = str(model) if model is not None else ""
            self.api_key = str(api_key) if api_key is not None else ""
            self.settings = settings if settings is not None else {} # Ensure settings is a dict
        except ValueError as ve:
            # In a real app, logger would be ideal here if available early.
            # For now, printing to stderr or raising.
            print(f"Error initializing LLMPreset '{name}': {ve}", file=sys.stderr) # Python's own logger might not be configured yet.
            raise # Re-raise to indicate failure to create the object properly

class StartupConfigDialog(QDialog):
    """Startup configuration dialog for LLM selection"""

    config_complete = pyqtSignal(dict) # Parameter is dict

    def __init__(self, parent: Optional[QWidget] = None): # Added Optional[QWidget] for parent type
        super().__init__(parent)
        self.setWindowTitle("AI Stock Analyzer - Configuration")
        self.setModal(True)
        self.setMinimumSize(800, 600)

        try:
            self.settings = QSettings("StockAnalyzer", "Settings") # QSettings is generally robust
        except Exception as e: # Should be extremely rare for QSettings constructor # pragma: no cover
            # Fallback if QSettings fails critically (e.g. permissions, disk full for temp files)
            print(f"Critical Error: Could not initialize QSettings: {e}. Using memory-only settings.", file=sys.stderr)
            # Provide a dummy QSettings object that doesn't persist, to allow dialog to function somewhat.
            self.settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "StockAnalyzer_Fallback", "Settings_Fallback")


        try:
            self.presets = self.load_presets()
        except Exception as e: # If load_presets fails, start with empty/default list # pragma: no cover
            print(f"Error loading presets: {e}. Starting with default presets.", file=sys.stderr)
            self.presets = self._get_default_presets() # Use a helper for defaults

        self.selected_preset: Optional[LLMPreset] = None # Type hint for clarity

        try:
            self.init_ui()
            self.load_last_config()
        except Exception as e: # Catch any broad error during UI init or last config load # pragma: no cover
            print(f"Error during StartupConfigDialog UI initialization: {e}", file=sys.stderr)
            # Show a simple QMessageBox as the UI might be partially broken
            QMessageBox.critical(self, "Dialog Initialization Error",
                                 f"Could not fully initialize the configuration dialog: {e}\n"
                                 "Please check console for details. Defaults will be used where possible.")
            # Attempt to continue with a potentially partially initialized UI

    def _get_default_presets(self) -> list[LLMPreset]:
        """Returns a list of default LLMPreset objects."""
        return [
            LLMPreset("Quick Start - OpenAI", "openai", "gpt-3.5-turbo", ""),
            LLMPreset("Premium - GPT-4", "openai", "gpt-4", ""),
            LLMPreset("Local - LM Studio", "lmstudio", "", "", {"url": "http://localhost:1234"})
        ]

    def init_ui(self):
        """Initialize UI"""
        try:
            layout = QVBoxLayout(self)
            layout.setSpacing(20)

            # Header
            header = QLabel("Welcome to AI Stock Analyzer")
            header.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    color: #3b82f6;
                    padding: 20px;
                }
            """)
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(header)

            # Main content area
            content_layout = QHBoxLayout()

            # Left panel - Preset list
            left_panel = self.create_left_panel() # Assuming this also has error handling or is robust
            content_layout.addWidget(left_panel, 1)

            # Right panel - Configuration
            self.config_stack = QStackedWidget()

            # Add configuration panels for each provider
            self.openai_config = self.create_openai_config()
            self.anthropic_config = self.create_anthropic_config()
            self.gemini_config = self.create_gemini_config()
            self.huggingface_config = self.create_huggingface_config()
            self.lmstudio_config = self.create_lmstudio_config()

            self.config_stack.addWidget(self.openai_config)
            self.config_stack.addWidget(self.anthropic_config)
            self.config_stack.addWidget(self.gemini_config)
            self.config_stack.addWidget(self.huggingface_config)
            self.config_stack.addWidget(self.lmstudio_config)

            content_layout.addWidget(self.config_stack, 2)
            layout.addLayout(content_layout)

            # Quick start options
            quick_start = QGroupBox("Quick Start Options")
            quick_layout = QVBoxLayout(quick_start)

            self.skip_market_data = QCheckBox("Skip initial market data download (faster startup)")
             # Safely get boolean value from settings
            skip_market_data_val = self.settings.value("quick_start/skip_market_data", True, type=bool)
            self.skip_market_data.setChecked(skip_market_data_val)
            quick_layout.addWidget(self.skip_market_data)

            self.demo_mode = QCheckBox("Demo mode (use mock data for testing)")
            demo_mode_val = self.settings.value("quick_start/demo_mode", False, type=bool)
            self.demo_mode.setChecked(demo_mode_val)
            quick_layout.addWidget(self.demo_mode)

            layout.addWidget(quick_start)

            # Buttons
            button_layout = QHBoxLayout()

            self.test_btn = QPushButton("Test Connection")
            self.test_btn.clicked.connect(self.test_connection)
            button_layout.addWidget(self.test_btn)

            button_layout.addStretch()

            self.save_preset_btn = QPushButton("Save as Preset")
            self.save_preset_btn.clicked.connect(self.save_preset)
            button_layout.addWidget(self.save_preset_btn)

            self.start_btn = QPushButton("Start Application")
            self.start_btn.setDefault(True)
            self.start_btn.clicked.connect(self.start_application)
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3b82f6;
                    color: white;
                    padding: 10px 30px;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                }
            """)
            button_layout.addWidget(self.start_btn)
            layout.addLayout(button_layout)

            # Apply dark theme (QSS parsing itself is usually robust)
            self.setStyleSheet("""
                QDialog { background-color: #13141a; }
                QGroupBox { background-color: #1a1b23; border: 1px solid #2a2b35; border-radius: 8px; padding: 10px; margin-top: 10px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 10px; }
                QLineEdit, QComboBox, QSpinBox { background-color: #13141a; border: 2px solid #2a2b35; padding: 8px; border-radius: 6px; color: #e8eaed; }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #3b82f6; }
                QPushButton { background-color: #374151; color: white; border: none; padding: 8px 16px; border-radius: 6px; }
                QPushButton:hover { background-color: #4b5563; }
                QLabel { color: #e8eaed; }
                QListWidget { background-color: #1a1b23; border: 1px solid #2a2b35; border-radius: 8px; }
                QListWidget::item { padding: 10px; border-radius: 6px; margin: 2px; }
                QListWidget::item:hover { background-color: rgba(59, 130, 246, 0.1); }
                QListWidget::item:selected { background-color: #3b82f6; }
                QCheckBox { color: #e8eaed; }
            """)
        except Exception as e: # Catch any error during UI element creation
            # This is a fallback if individual component creation methods don't handle their own errors.
            print(f"Critical error setting up StartupConfigDialog UI: {e}", file=sys.stderr)
            # Fallback to a very simple UI if complex setup fails.
            fallback_layout = QVBoxLayout(self)
            fallback_layout.addWidget(QLabel("Error: Could not load configuration UI. Please check logs."))
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(self.reject) # Close dialog on error
            fallback_layout.addWidget(ok_button)


    def create_left_panel(self) -> QWidget:
        """Create left panel with presets"""
        panel = QGroupBox("LLM Provider Selection")
        layout = QVBoxLayout(panel)

        # Provider type selector
        self.provider_combo = QComboBox()
        providers = [
            ("OpenAI (GPT-4)", "openai"),
            ("Anthropic (Claude)", "anthropic"),
            ("Google Gemini", "gemini"),
            ("HuggingFace", "huggingface"),
            ("LM Studio (Local)", "lmstudio")
        ]

        for name, key in providers:
            self.provider_combo.addItem(name, key)

        self.provider_combo.currentIndexChanged.connect(self.provider_changed)
        layout.addWidget(QLabel("Select Provider:"))
        layout.addWidget(self.provider_combo)

        # Presets list
        layout.addWidget(QLabel("Saved Presets:"))
        self.preset_list = QListWidget()
        self.preset_list.itemClicked.connect(self.preset_selected)

        self.update_preset_list()
        layout.addWidget(self.preset_list)

        # Delete preset button
        self.delete_preset_btn = QPushButton("Delete Selected Preset")
        self.delete_preset_btn.clicked.connect(self.delete_preset)
        layout.addWidget(self.delete_preset_btn)

        return panel

    def create_openai_config(self) -> QWidget:
        """Create OpenAI configuration panel"""
        widget = QWidget()
        layout = QFormLayout(widget)

        info = QLabel("Configure OpenAI GPT-4 for advanced analysis")
        info.setWordWrap(True)
        layout.addRow(info)

        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key.setPlaceholderText("sk-...")
        layout.addRow("API Key:", self.openai_key)

        self.openai_model = QComboBox()
        self.openai_model.addItems(["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"])
        layout.addRow("Model:", self.openai_model)

        help_text = QLabel('<a href="https://platform.openai.com/api-keys">Get API Key</a>')
        help_text.setOpenExternalLinks(True)
        layout.addRow(help_text)

        return widget

    def create_anthropic_config(self) -> QWidget:
        """Create Anthropic configuration panel"""
        widget = QWidget()
        layout = QFormLayout(widget)

        info = QLabel("Configure Anthropic Claude for thoughtful analysis")
        info.setWordWrap(True)
        layout.addRow(info)

        self.anthropic_key = QLineEdit()
        self.anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.anthropic_key.setPlaceholderText("sk-ant-...")
        layout.addRow("API Key:", self.anthropic_key)

        self.anthropic_model = QComboBox()
        self.anthropic_model.addItems(["claude-3-opus-20240229", "claude-3-sonnet-20240229"])
        layout.addRow("Model:", self.anthropic_model)

        help_text = QLabel('<a href="https://console.anthropic.com/">Get API Key</a>')
        help_text.setOpenExternalLinks(True)
        layout.addRow(help_text)

        return widget

    def create_gemini_config(self) -> QWidget:
        """Create Google Gemini configuration panel"""
        widget = QWidget()
        layout = QFormLayout(widget)

        info = QLabel("Configure Google Gemini for fast analysis")
        info.setWordWrap(True)
        layout.addRow(info)

        self.gemini_key = QLineEdit()
        self.gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key.setPlaceholderText("AI...")
        layout.addRow("API Key:", self.gemini_key)

        help_text = QLabel('<a href="https://makersuite.google.com/app/apikey">Get API Key</a>')
        help_text.setOpenExternalLinks(True)
        layout.addRow(help_text)

        return widget

    def create_huggingface_config(self) -> QWidget:
        """Create HuggingFace configuration panel"""
        widget = QWidget()
        layout = QFormLayout(widget)

        info = QLabel("Configure HuggingFace for open-source models")
        info.setWordWrap(True)
        layout.addRow(info)

        self.huggingface_key = QLineEdit()
        self.huggingface_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.huggingface_key.setPlaceholderText("hf_...")
        layout.addRow("API Token:", self.huggingface_key)

        self.huggingface_model = QComboBox()
        self.huggingface_model.setEditable(True)
        self.huggingface_model.addItems([
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Llama-2-70b-chat-hf",
            "tiiuae/falcon-40b-instruct"
        ])
        layout.addRow("Model:", self.huggingface_model)

        help_text = QLabel('<a href="https://huggingface.co/settings/tokens">Get API Token</a>')
        help_text.setOpenExternalLinks(True)
        layout.addRow(help_text)

        return widget

    def create_lmstudio_config(self) -> QWidget:
        """Create LM Studio configuration panel"""
        widget = QWidget()
        layout = QFormLayout(widget)

        info = QLabel("Configure LM Studio for local LLM inference (no API key required)")
        info.setWordWrap(True)
        layout.addRow(info)

        self.lmstudio_url = QLineEdit()
        self.lmstudio_url.setText("http://localhost:1234")
        layout.addRow("Server URL:", self.lmstudio_url)

        self.lmstudio_model = QLineEdit()
        self.lmstudio_model.setPlaceholderText("Optional - uses loaded model")
        layout.addRow("Model Name:", self.lmstudio_model)

        help_text = QLabel("Make sure LM Studio is running with a model loaded")
        help_text.setStyleSheet("color: #8b92a1;")
        layout.addRow(help_text)

        return widget

    def provider_changed(self, index: int):
        """Handle provider change"""
        self.config_stack.setCurrentIndex(index)

    def preset_selected(self, item: Optional[QListWidgetItem]): # Allow item to be None
        """Handle preset selection"""
        if not item: # If no item is selected (e.g., list is cleared)
            self.selected_preset = None
            # Optionally, clear input fields or reset to a default state
            return

        preset_name = item.text()
        try:
            preset = next((p for p in self.presets if p.name == preset_name), None)
        except Exception as e: # pragma: no cover
            print(f"Error finding preset '{preset_name}': {e}", file=sys.stderr)
            self.selected_preset = None
            return

        if preset:
            self.selected_preset = preset
            provider_map = {"openai": 0, "anthropic": 1, "gemini": 2, "huggingface": 3, "lmstudio": 4}

            try:
                if preset.provider in provider_map:
                    self.provider_combo.setCurrentIndex(provider_map[preset.provider])

                    # Set configuration safely, checking if UI elements exist
                    if preset.provider == "openai" and hasattr(self, 'openai_key') and hasattr(self, 'openai_model'):
                        self.openai_key.setText(preset.api_key or "")
                        if preset.model:
                            index = self.openai_model.findText(preset.model)
                            if index >= 0: self.openai_model.setCurrentIndex(index)
                            else: self.openai_model.setCurrentText(preset.model) # Add if not found
                    elif preset.provider == "anthropic" and hasattr(self, 'anthropic_key') and hasattr(self, 'anthropic_model'):
                        self.anthropic_key.setText(preset.api_key or "")
                        if preset.model:
                            index = self.anthropic_model.findText(preset.model)
                            if index >= 0: self.anthropic_model.setCurrentIndex(index)
                            else: self.anthropic_model.setCurrentText(preset.model)
                    elif preset.provider == "gemini" and hasattr(self, 'gemini_key'):
                        self.gemini_key.setText(preset.api_key or "")
                    elif preset.provider == "huggingface" and hasattr(self, 'huggingface_key') and hasattr(self, 'huggingface_model'):
                        self.huggingface_key.setText(preset.api_key or "")
                        if preset.model: self.huggingface_model.setCurrentText(preset.model)
                    elif preset.provider == "lmstudio" and hasattr(self, 'lmstudio_url') and hasattr(self, 'lmstudio_model'):
                        self.lmstudio_url.setText(preset.settings.get("url", "http://localhost:1234"))
                        self.lmstudio_model.setText(preset.model or "")
                else: # pragma: no cover
                    print(f"Warning: Preset provider '{preset.provider}' not in UI map.", file=sys.stderr)
            except AttributeError as ae: # pragma: no cover
                 print(f"UI element missing when selecting preset '{preset_name}': {ae}", file=sys.stderr)
            except Exception as e: # pragma: no cover
                 print(f"Error applying preset '{preset_name}': {e}", file=sys.stderr)
        else: # pragma: no cover
            self.selected_preset = None # Preset not found in self.presets
            print(f"Warning: Preset '{preset_name}' not found in loaded presets.", file=sys.stderr)


    def test_connection(self):
        """Test LLM connection (currently a placeholder)"""
        try:
            config = self.get_current_config()

            if not self.validate_config(config): # validate_config shows its own QMessageBox
                return

            # Actual test logic would go here.
            # This would involve:
            # 1. Dynamically creating an LLMProvider instance based on `config`.
            #    from main_window import MainWindow # To access create_llm_provider or similar logic
            #    # temp_provider = MainWindow.create_llm_provider_from_config(config) # Hypothetical static method
            # 2. Calling a lightweight method on the provider, e.g., provider.ping() or a simple completion.
            # 3. Handling exceptions from that call (network, API key, etc.).

            # For now, as it's a placeholder:
            QMessageBox.information(self, "Connection Test",
                                    "This is a placeholder for LLM connection testing.\n"
                                    "In a real version, this would verify your API key and settings.")
            # Simulate success for now
            # QMessageBox.information(self, "Success", "LLM connection successful! (Simulated)")
        except Exception as e: # pragma: no cover
            logger.exception("Error during (placeholder) connection test setup.") # Use actual logger if available globally
            QMessageBox.critical(self, "Test Error", f"Could not prepare for connection test: {e}")


    def save_preset(self):
        """Save current configuration as preset"""
        try:
            config = self.get_current_config()

            if not self.validate_config(config): # Shows its own error message
                return

            name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")

            if ok and name.strip():
                name = name.strip()
                # Check if preset name already exists
                if any(p.name.lower() == name.lower() for p in self.presets):
                    overwrite = QMessageBox.question(self, "Preset Exists",
                                                     f"Preset '{name}' already exists. Overwrite it?",
                                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if overwrite == QMessageBox.StandardButton.No:
                        return
                    # Remove existing to overwrite
                    self.presets = [p for p in self.presets if p.name.lower() != name.lower()]

                preset = LLMPreset(
                    name=name,
                    provider=config["provider"],
                    model=config.get("model", ""),
                    api_key=config.get("api_key", ""),
                    settings=config.get("settings", {})
                )
                self.presets.append(preset)
                self.save_presets() # Handles its own errors
                self.update_preset_list() # Handles its own errors
                QMessageBox.information(self, "Success", f"Preset '{name}' saved successfully!")
            elif ok and not name.strip(): # pragma: no cover
                 QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
        except Exception as e: # pragma: no cover
            logger.exception("Error saving preset.")
            QMessageBox.critical(self, "Save Preset Error", f"Could not save preset: {e}")


    def delete_preset(self):
        """Delete selected preset"""
        try:
            current_item = self.preset_list.currentItem()
            if not current_item:
                QMessageBox.information(self, "Delete Preset", "Please select a preset to delete.")
                return

            preset_name = current_item.text()

            reply = QMessageBox.question(
                self, "Delete Preset",
                f"Are you sure you want to delete preset '{preset_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No # Default button
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.presets = [p for p in self.presets if p.name != preset_name]
                self.save_presets() # Handles its own errors
                self.update_preset_list() # Handles its own errors
                QMessageBox.information(self, "Preset Deleted", f"Preset '{preset_name}' deleted.")
        except Exception as e: # pragma: no cover
            logger.exception("Error deleting preset.")
            QMessageBox.critical(self, "Delete Preset Error", f"Could not delete preset: {e}")


    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration from UI fields"""
        try:
            provider_data = self.provider_combo.currentData() # .currentData() holds the key like "openai"
            provider = provider_data if isinstance(provider_data, str) else "openai" # Fallback

            config: Dict[str, Any] = { # Ensure type hint for config
                "provider": provider,
                "skip_market_data": self.skip_market_data.isChecked() if hasattr(self, 'skip_market_data') else False,
                "demo_mode": self.demo_mode.isChecked() if hasattr(self, 'demo_mode') else False,
                "api_key": "", # Default empty
                "model": "",   # Default empty
                "settings": {} # Default empty
            }

            if provider == "openai" and hasattr(self, 'openai_key') and hasattr(self, 'openai_model'):
                config["api_key"] = self.openai_key.text().strip()
                config["model"] = self.openai_model.currentText()
            elif provider == "anthropic" and hasattr(self, 'anthropic_key') and hasattr(self, 'anthropic_model'):
                config["api_key"] = self.anthropic_key.text().strip()
                config["model"] = self.anthropic_model.currentText()
            elif provider == "gemini" and hasattr(self, 'gemini_key'):
                config["api_key"] = self.gemini_key.text().strip()
            elif provider == "huggingface" and hasattr(self, 'huggingface_key') and hasattr(self, 'huggingface_model'):
                config["api_key"] = self.huggingface_key.text().strip()
                config["model"] = self.huggingface_model.currentText()
            elif provider == "lmstudio" and hasattr(self, 'lmstudio_url') and hasattr(self, 'lmstudio_model'):
                config["settings"] = {"url": self.lmstudio_url.text().strip()}
                config["model"] = self.lmstudio_model.text().strip()
                # For LM Studio, api_key field in config remains empty as URL is in settings.

            return config
        except AttributeError as ae: # pragma: no cover
            print(f"Error: UI element missing in get_current_config: {ae}. This might indicate an incomplete UI initialization.", file=sys.stderr)
            # Return a very basic default config to prevent further crashes
            return {"provider": "openai", "api_key": "", "model": "", "settings": {}, "skip_market_data": True, "demo_mode": False}
        except Exception as e: # pragma: no cover
            print(f"Unexpected error in get_current_config: {e}", file=sys.stderr)
            return {"provider": "openai", "api_key": "", "model": "", "settings": {}, "skip_market_data": True, "demo_mode": False}


    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        try:
            provider = config.get("provider")
            if not provider: # Should always be set by get_current_config
                QMessageBox.warning(self, "Configuration Error", "LLM Provider is not selected.")
                return False

            if provider in ["openai", "anthropic", "gemini", "huggingface"]:
                if not config.get("api_key", "").strip(): # Check if API key is empty or just whitespace
                    QMessageBox.warning(self, "Missing API Key",
                                      f"Please enter an API key for {provider.title()}.")
                    return False
            elif provider == "lmstudio":
                url = config.get("settings", {}).get("url", "").strip()
                if not url:
                    QMessageBox.warning(self, "Missing URL",
                                      "Please enter the LM Studio server URL.")
                    return False
                if not (url.startswith("http://") or url.startswith("https://")): # Basic URL validation
                    QMessageBox.warning(self, "Invalid URL",
                                      "LM Studio server URL must start with http:// or https://.")
                    return False
            else: # Unknown provider from get_current_config, should not happen # pragma: no cover
                QMessageBox.critical(self, "Internal Error", f"Unknown provider '{provider}' encountered during validation.")
                return False
            return True
        except Exception as e: # pragma: no cover
            logger.exception(f"Error validating configuration: {e}")
            QMessageBox.critical(self, "Validation Error", f"An unexpected error occurred during configuration validation: {e}")
            return False


    def start_application(self):
        """Start the application with current configuration"""
        try:
            config = self.get_current_config()

            if not self.validate_config(config):
                return

            self.save_last_config(config) # Save for next session

            self.config_complete.emit(config) # Emit the validated config
            self.accept() # Close the dialog
        except Exception as e: # pragma: no cover
            logger.exception("Error during start_application sequence.")
            QMessageBox.critical(self, "Application Start Error", f"Could not start application: {e}")


    def load_presets(self) -> list[LLMPreset]: # Return type hint
        """Load saved presets"""
        # Get script directory robustly
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:  # __file__ is not defined (e.g. in frozen app or interactive) # pragma: no cover
            script_dir = os.getcwd()
        presets_file = os.path.join(script_dir, "llm_presets.json")


        if os.path.exists(presets_file):
            try:
                with open(presets_file, 'r', encoding='utf-8') as f: # Specify encoding
                    raw_data_list = json.load(f)

                loaded_presets = []
                if isinstance(raw_data_list, list):
                    for i, p_data in enumerate(raw_data_list):
                        if isinstance(p_data, dict):
                            try:
                                preset = LLMPreset(
                                    name=str(p_data.get("name", f"Unnamed Preset {i+1}")),
                                    provider=str(p_data.get("provider", "openai")), # Default provider if missing
                                    model=str(p_data.get("model", "")),
                                    api_key=str(p_data.get("api_key", "")),
                                    settings=p_data.get("settings") if isinstance(p_data.get("settings"), dict) else {}
                                )
                                loaded_presets.append(preset)
                            except (TypeError, ValueError) as e_item: # Error instantiating LLMPreset from item data
                                print(f"Warning: Skipping malformed preset item #{i}: {p_data}. Error: {e_item}", file=sys.stderr)
                        else: # pragma: no cover
                             print(f"Warning: Preset item #{i} is not a dictionary, skipping: {p_data}", file=sys.stderr)
                    return loaded_presets
                else: # pragma: no cover
                    print(f"Warning: Presets file '{presets_file}' does not contain a list. Using defaults.", file=sys.stderr)
            except (IOError, OSError) as e: # pragma: no cover
                print(f"Error: Could not read presets file '{presets_file}': {e}", file=sys.stderr)
            except json.JSONDecodeError as e: # pragma: no cover
                print(f"Error: Invalid JSON in presets file '{presets_file}': {e}", file=sys.stderr)
            except Exception as e: # Catch any other unexpected error # pragma: no cover
                print(f"Error: Unexpected error loading presets from '{presets_file}': {e}", file=sys.stderr)

        # Fallback to default presets if file doesn't exist or fails to load/parse
        return self._get_default_presets()


    def save_presets(self):
        """Save presets to file"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: # pragma: no cover
            script_dir = os.getcwd()
        presets_file = os.path.join(script_dir, "llm_presets.json")

        try:
            data_to_save = []
            for p in self.presets:
                if isinstance(p, LLMPreset): # Ensure it's an LLMPreset object
                    data_to_save.append({
                        "name": p.name, "provider": p.provider, "model": p.model,
                        "api_key": p.api_key, "settings": p.settings
                    })
                else: # pragma: no cover
                    print(f"Warning: Attempted to save non-LLMPreset object to presets: {p}", file=sys.stderr)


            with open(presets_file, 'w', encoding='utf-8') as f: # Specify encoding
                json.dump(data_to_save, f, indent=2)
        except (IOError, OSError) as e: # pragma: no cover
            print(f"Error: Could not write presets to file '{presets_file}': {e}", file=sys.stderr)
            QMessageBox.critical(self, "Save Error", f"Could not save presets: {e}")
        except TypeError as te: # If data_to_save is not serializable (shouldn't happen with this structure) # pragma: no cover
            print(f"Error: Data for presets is not JSON serializable: {te}", file=sys.stderr)
            QMessageBox.critical(self, "Save Error", "Could not serialize preset data.")
        except Exception as e: # pragma: no cover
            print(f"Error: Unexpected error saving presets: {e}", file=sys.stderr)
            QMessageBox.critical(self, "Save Error", f"An unexpected error occurred while saving presets: {e}")


    def update_preset_list(self):
        """Update preset list display"""
        try:
            self.preset_list.clear()
            if not self.presets: # If presets list is empty (e.g. after error in load_presets)
                item = QListWidgetItem("No presets available")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable) # Make it unselectable
                self.preset_list.addItem(item)
            else:
                for preset in self.presets:
                    if isinstance(preset, LLMPreset): # Ensure it's an LLMPreset object
                        item = QListWidgetItem(preset.name)
                        self.preset_list.addItem(item)
                    else: # pragma: no cover
                        print(f"Warning: Invalid object in presets list: {preset}", file=sys.stderr)
        except AttributeError: # If self.preset_list doesn't exist (UI init failed) # pragma: no cover
            print("Error: Preset list UI element not available.", file=sys.stderr)
        except Exception as e: # pragma: no cover
            print(f"Error updating preset list UI: {e}", file=sys.stderr)


    def load_last_config(self):
        """Load last used configuration from QSettings"""
        try:
            provider = str(self.settings.value("startup_provider", "")) # Default to empty string
            if provider:
                provider_map = {
                    "openai": 0, "anthropic": 1, "gemini": 2,
                    "huggingface": 3, "lmstudio": 4
                }
                if provider in provider_map and hasattr(self, 'provider_combo'):
                    self.provider_combo.setCurrentIndex(provider_map[provider])
                    # After setting provider, also load its specific settings
                    self.provider_changed(provider_map[provider]) # Trigger update of config stack

                    # Now, load specific fields for this provider
                    if provider == "openai" and hasattr(self, 'openai_key') and hasattr(self, 'openai_model'):
                        self.openai_key.setText(str(self.settings.value("openai_key", "")))
                        self.openai_model.setCurrentText(str(self.settings.value("openai_model", "gpt-4")))
                    elif provider == "anthropic" and hasattr(self, 'anthropic_key') and hasattr(self, 'anthropic_model'):
                        self.anthropic_key.setText(str(self.settings.value("anthropic_key", "")))
                        self.anthropic_model.setCurrentText(str(self.settings.value("anthropic_model", "")))
                    elif provider == "gemini" and hasattr(self, 'gemini_key'):
                        self.gemini_key.setText(str(self.settings.value("gemini_key", "")))
                    elif provider == "huggingface" and hasattr(self, 'huggingface_key') and hasattr(self, 'huggingface_model'):
                        self.huggingface_key.setText(str(self.settings.value("huggingface_key", "")))
                        self.huggingface_model.setCurrentText(str(self.settings.value("huggingface_model", "")))
                    elif provider == "lmstudio" and hasattr(self, 'lmstudio_url') and hasattr(self, 'lmstudio_model'):
                        self.lmstudio_url.setText(str(self.settings.value("lmstudio_url", "http://localhost:1234")))
                        self.lmstudio_model.setText(str(self.settings.value("lmstudio_model", "")))

            # Load quick start options
            if hasattr(self, 'skip_market_data'):
                self.skip_market_data.setChecked(self.settings.value("quick_start/skip_market_data", True, type=bool))
            if hasattr(self, 'demo_mode'):
                self.demo_mode.setChecked(self.settings.value("quick_start/demo_mode", False, type=bool))

        except Exception as e: # pragma: no cover
            print(f"Error loading last configuration: {e}", file=sys.stderr)
            QMessageBox.warning(self, "Load Settings Error", f"Could not load last used settings: {e}")


    def save_last_config(self, config: Dict[str, Any]):
        """Save configuration for next startup to QSettings"""
        try:
            provider = config.get("provider")
            if not provider: return # Don't save if provider is missing

            self.settings.setValue("startup_provider", provider)

            # Save provider-specific settings
            if provider == "openai":
                self.settings.setValue("openai_key", config.get("api_key", ""))
                self.settings.setValue("openai_model", config.get("model", "gpt-4"))
            elif provider == "anthropic":
                self.settings.setValue("anthropic_key", config.get("api_key", ""))
                self.settings.setValue("anthropic_model", config.get("model", ""))
            elif provider == "gemini":
                self.settings.setValue("gemini_key", config.get("api_key", ""))
            elif provider == "huggingface":
                self.settings.setValue("huggingface_key", config.get("api_key", ""))
                self.settings.setValue("huggingface_model", config.get("model", ""))
            elif provider == "lmstudio":
                self.settings.setValue("lmstudio_url", config.get("settings", {}).get("url", "http://localhost:1234"))
                self.settings.setValue("lmstudio_model", config.get("model", ""))

            # Save general settings (which might be also used by main_window's QSettings)
            self.settings.setValue("llm_provider", provider.title()) # Save the display name used by main_window

            # Save quick start options
            self.settings.setValue("quick_start/skip_market_data", config.get("skip_market_data", True))
            self.settings.setValue("quick_start/demo_mode", config.get("demo_mode", False))

            self.settings.sync() # Persist immediately
        except Exception as e: # pragma: no cover
            logger.exception(f"Error saving last configuration: {e}")
            # Non-critical, don't show QMessageBox as this is usually called during shutdown/accept.
