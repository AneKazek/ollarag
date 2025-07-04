import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ui.app

if __name__ == "__main__":
    ui.app.main()