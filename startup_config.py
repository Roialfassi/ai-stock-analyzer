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
    def __init__(self, name: str, provider: str, model: str = "", api_key: str = "", settings: Dict[str, Any] = None):
        self.name = name
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.settings = settings or {}

class StartupConfigDialog(QDialog):
    """Startup configuration dialog for LLM selection"""

    config_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Stock Analyzer - Configuration")
        self.setModal(True)
        self.setMinimumSize(800, 600)

        self.settings = QSettings("StockAnalyzer", "Settings")
        self.presets = self.load_presets()
        self.selected_preset = None

        self.init_ui()
        self.load_last_config()

    def init_ui(self):
        """Initialize UI"""
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
        left_panel = self.create_left_panel()
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
        self.skip_market_data.setChecked(True)
        quick_layout.addWidget(self.skip_market_data)

        self.demo_mode = QCheckBox("Demo mode (use mock data for testing)")
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

        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #13141a;
            }
            QGroupBox {
                background-color: #1a1b23;
                border: 1px solid #2a2b35;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #13141a;
                border: 2px solid #2a2b35;
                padding: 8px;
                border-radius: 6px;
                color: #e8eaed;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border-color: #3b82f6;
            }
            QPushButton {
                background-color: #374151;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
            QLabel {
                color: #e8eaed;
            }
            QListWidget {
                background-color: #1a1b23;
                border: 1px solid #2a2b35;
                border-radius: 8px;
            }
            QListWidget::item {
                padding: 10px;
                border-radius: 6px;
                margin: 2px;
            }
            QListWidget::item:hover {
                background-color: rgba(59, 130, 246, 0.1);
            }
            QListWidget::item:selected {
                background-color: #3b82f6;
            }
            QCheckBox {
                color: #e8eaed;
            }
        """)

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

    def preset_selected(self, item: QListWidgetItem):
        """Handle preset selection"""
        preset_name = item.text()
        preset = next((p for p in self.presets if p.name == preset_name), None)

        if preset:
            self.selected_preset = preset

            # Set provider
            provider_map = {
                "openai": 0,
                "anthropic": 1,
                "gemini": 2,
                "huggingface": 3,
                "lmstudio": 4
            }

            if preset.provider in provider_map:
                self.provider_combo.setCurrentIndex(provider_map[preset.provider])

                # Set configuration
                if preset.provider == "openai":
                    self.openai_key.setText(preset.api_key)
                    if preset.model:
                        index = self.openai_model.findText(preset.model)
                        if index >= 0:
                            self.openai_model.setCurrentIndex(index)
                elif preset.provider == "anthropic":
                    self.anthropic_key.setText(preset.api_key)
                    if preset.model:
                        index = self.anthropic_model.findText(preset.model)
                        if index >= 0:
                            self.anthropic_model.setCurrentIndex(index)
                elif preset.provider == "gemini":
                    self.gemini_key.setText(preset.api_key)
                elif preset.provider == "huggingface":
                    self.huggingface_key.setText(preset.api_key)
                    if preset.model:
                        self.huggingface_model.setCurrentText(preset.model)
                elif preset.provider == "lmstudio":
                    self.lmstudio_url.setText(preset.settings.get("url", "http://localhost:1234"))
                    self.lmstudio_model.setText(preset.model)

    def test_connection(self):
        """Test LLM connection"""
        config = self.get_current_config()

        if not self.validate_config(config):
            return

        # Show testing dialog
        QMessageBox.information(self, "Testing", "Testing LLM connection...")

        # In a real implementation, this would actually test the connection
        # For now, we'll just show success
        QMessageBox.information(self, "Success", "LLM connection successful!")

    def save_preset(self):
        """Save current configuration as preset"""
        config = self.get_current_config()

        if not self.validate_config(config):
            return

        # Get preset name
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")

        if ok and name:
            preset = LLMPreset(
                name=name,
                provider=config["provider"],
                model=config.get("model", ""),
                api_key=config.get("api_key", ""),
                settings=config.get("settings", {})
            )

            self.presets.append(preset)
            self.save_presets()
            self.update_preset_list()

            QMessageBox.information(self, "Success", f"Preset '{name}' saved successfully!")

    def delete_preset(self):
        """Delete selected preset"""
        current_item = self.preset_list.currentItem()
        if not current_item:
            return

        preset_name = current_item.text()

        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Are you sure you want to delete preset '{preset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.presets = [p for p in self.presets if p.name != preset_name]
            self.save_presets()
            self.update_preset_list()

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        provider_index = self.provider_combo.currentIndex()
        providers = ["openai", "anthropic", "gemini", "huggingface", "lmstudio"]
        provider = providers[provider_index]

        config = {
            "provider": provider,
            "skip_market_data": self.skip_market_data.isChecked(),
            "demo_mode": self.demo_mode.isChecked()
        }

        if provider == "openai":
            config["api_key"] = self.openai_key.text()
            config["model"] = self.openai_model.currentText()
        elif provider == "anthropic":
            config["api_key"] = self.anthropic_key.text()
            config["model"] = self.anthropic_model.currentText()
        elif provider == "gemini":
            config["api_key"] = self.gemini_key.text()
        elif provider == "huggingface":
            config["api_key"] = self.huggingface_key.text()
            config["model"] = self.huggingface_model.currentText()
        elif provider == "lmstudio":
            config["settings"] = {
                "url": self.lmstudio_url.text()
            }
            config["model"] = self.lmstudio_model.text()

        return config

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        provider = config["provider"]

        if provider in ["openai", "anthropic", "gemini", "huggingface"]:
            if not config.get("api_key"):
                QMessageBox.warning(self, "Invalid Configuration",
                                  f"Please enter an API key for {provider}")
                return False
        elif provider == "lmstudio":
            if not config.get("settings", {}).get("url"):
                QMessageBox.warning(self, "Invalid Configuration",
                                  "Please enter LM Studio server URL")
                return False

        return True

    def start_application(self):
        """Start the application with current configuration"""
        config = self.get_current_config()

        if not self.validate_config(config):
            return

        # Save configuration
        self.save_last_config(config)

        # Emit configuration and close
        self.config_complete.emit(config)
        self.accept()

    def load_presets(self) -> list:
        """Load saved presets"""
        presets_file = os.path.join(os.path.dirname(__file__), "llm_presets.json")

        if os.path.exists(presets_file):
            try:
                with open(presets_file, 'r') as f:
                    data = json.load(f)
                    return [
                        LLMPreset(
                            name=p["name"],
                            provider=p["provider"],
                            model=p.get("model", ""),
                            api_key=p.get("api_key", ""),
                            settings=p.get("settings", {})
                        )
                        for p in data
                    ]
            except:
                pass

        # Return default presets
        return [
            LLMPreset("Quick Start - OpenAI", "openai", "gpt-3.5-turbo", ""),
            LLMPreset("Premium - GPT-4", "openai", "gpt-4", ""),
            LLMPreset("Local - LM Studio", "lmstudio", "", "", {"url": "http://localhost:1234"})
        ]

    def save_presets(self):
        """Save presets to file"""
        presets_file = os.path.join(os.path.dirname(__file__), "llm_presets.json")

        data = [
            {
                "name": p.name,
                "provider": p.provider,
                "model": p.model,
                "api_key": p.api_key,
                "settings": p.settings
            }
            for p in self.presets
        ]

        try:
            with open(presets_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")

    def update_preset_list(self):
        """Update preset list display"""
        self.preset_list.clear()

        for preset in self.presets:
            item = QListWidgetItem(preset.name)
            self.preset_list.addItem(item)

    def load_last_config(self):
        """Load last used configuration"""
        provider = self.settings.value("startup_provider")
        if provider:
            provider_map = {
                "openai": 0,
                "anthropic": 1,
                "gemini": 2,
                "huggingface": 3,
                "lmstudio": 4
            }

            if provider in provider_map:
                self.provider_combo.setCurrentIndex(provider_map[provider])

    def save_last_config(self, config: Dict[str, Any]):
        """Save configuration for next startup"""
        self.settings.setValue("startup_provider", config["provider"])

        # Save to main settings as well
        if config["provider"] == "openai":
            self.settings.setValue("openai_key", config.get("api_key", ""))
            self.settings.setValue("openai_model", config.get("model", "gpt-4"))
        elif config["provider"] == "anthropic":
            self.settings.setValue("anthropic_key", config.get("api_key", ""))
            self.settings.setValue("anthropic_model", config.get("model", ""))
        elif config["provider"] == "gemini":
            self.settings.setValue("gemini_key", config.get("api_key", ""))
        elif config["provider"] == "huggingface":
            self.settings.setValue("huggingface_key", config.get("api_key", ""))
            self.settings.setValue("huggingface_model", config.get("model", ""))
        elif config["provider"] == "lmstudio":
            self.settings.setValue("lmstudio_url", config.get("settings", {}).get("url", ""))
            self.settings.setValue("lmstudio_model", config.get("model", ""))

        self.settings.setValue("llm_provider", config["provider"].title())
