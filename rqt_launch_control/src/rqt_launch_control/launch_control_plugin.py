"""rqt plugin entry point — thin wrapper around LaunchControlWidget."""

from qt_gui.plugin import Plugin
from .launch_control_widget import LaunchControlWidget


class LaunchControlPlugin(Plugin):

    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('LaunchControlPlugin')
        self._widget = LaunchControlWidget(context)
        context.add_widget(self._widget)

    def shutdown_plugin(self):
        self._widget.shutdown()

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass
