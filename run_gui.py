from __future__ import annotations

import os

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from habituation_analysis.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
