from __future__ import annotations

import sys

from gui import AppWindow, create_application


def main() -> int:
    app = create_application()
    window = AppWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
