"""rqt plugin entry point — thin wrapper around GGViewerWidget."""

from qt_gui.plugin import Plugin
from .gg_viewer_widget import GGViewerWidget


class GGViewerPlugin(Plugin):

    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('GGViewerPlugin')
        self._widget = GGViewerWidget(context)
        context.add_widget(self._widget)

    def shutdown_plugin(self):
        self._widget.shutdown()

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass
