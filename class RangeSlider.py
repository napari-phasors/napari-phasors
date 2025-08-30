class RangeSlider(QWidget):
    """A custom range slider with two handles on a single track."""

    rangeChanged = Signal(int, int)

    def __init__(self, min_val=0, max_val=100, initial_min=0, initial_max=100):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.min_handle = initial_min
        self.max_handle = initial_max
        self.handle_radius = 18
        self.track_height = 7
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)
        self.dragging = None  # Track which handle is being dragged

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate dimensions
        width = self.width() - 2 * self.handle_radius
        height = self.height()
        track_y = (height - self.track_height) // 2

        # Draw track background
        track_rect = QRectF(
            self.handle_radius, track_y, width, self.track_height
        )
        painter.fillRect(track_rect, QColor(64, 64, 64))

        # Calculate handle positions
        range_size = self.max_val - self.min_val
        if range_size > 0:
            min_pos = (self.min_handle - self.min_val) / range_size * width
            max_pos = (self.max_handle - self.min_val) / range_size * width
        else:
            min_pos = max_pos = 0

        # Draw active track (between handles)
        if max_pos > min_pos:
            active_rect = QRectF(
                self.handle_radius + min_pos,
                track_y,
                max_pos - min_pos,
                self.track_height,
            )
            painter.fillRect(active_rect, QColor(120, 120, 120))

        # Draw handles
        painter.setBrush(QBrush(QColor(155, 155, 155)))  # Full grey color
        painter.setPen(QPen(QColor(155, 155, 155)))

        # Min handle (left)
        painter.drawEllipse(
            int(min_pos + self.handle_radius - self.handle_radius // 2),
            int(height // 2 - self.handle_radius // 2),
            self.handle_radius,
            self.handle_radius,
        )

        # Max handle (right)
        painter.drawEllipse(
            int(max_pos + self.handle_radius - self.handle_radius // 2),
            int(height // 2 - self.handle_radius // 2),
            self.handle_radius,
            self.handle_radius,
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._handle_mouse_press(event)

    def mouseMoveEvent(self, event):
        if self.dragging is not None:
            self._handle_mouse_move(event)

    def mouseReleaseEvent(self, event):
        self.dragging = None

    def _handle_mouse_press(self, event):
        width = self.width() - 2 * self.handle_radius
        range_size = self.max_val - self.min_val

        if range_size == 0:
            return

        min_pos = (
            self.min_handle - self.min_val
        ) / range_size * width + self.handle_radius
        max_pos = (
            self.max_handle - self.min_val
        ) / range_size * width + self.handle_radius

        # Check which handle is closer
        dist_to_min = abs(event.x() - min_pos)
        dist_to_max = abs(event.x() - max_pos)

        if dist_to_min <= self.handle_radius and dist_to_min <= dist_to_max:
            self.dragging = 'min'
        elif dist_to_max <= self.handle_radius:
            self.dragging = 'max'

    def _handle_mouse_move(self, event):
        width = self.width() - 2 * self.handle_radius
        range_size = self.max_val - self.min_val

        if range_size == 0:
            return

        # Calculate new value based on mouse position
        pos_fraction = max(0, min(1, (event.x() - self.handle_radius) / width))
        new_value = int(self.min_val + pos_fraction * range_size)

        if self.dragging == 'min':
            self.min_handle = min(new_value, self.max_handle - 1)
        elif self.dragging == 'max':
            self.max_handle = max(new_value, self.min_handle + 1)

        self.update()
        self.rangeChanged.emit(self.min_handle, self.max_handle)

    def set_range(self, min_val, max_val):
        """Set the range of the slider."""
        self.min_val = min_val
        self.max_val = max_val
        self.update()

    def get_values(self):
        """Get current min and max values."""
        return self.min_handle, self.max_handle

    def set_values(self, min_val, max_val):
        """Set current min and max values."""
        self.min_handle = min_val
        self.max_handle = max_val
        self.update()