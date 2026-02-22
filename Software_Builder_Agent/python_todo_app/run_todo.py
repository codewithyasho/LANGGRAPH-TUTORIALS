#!/usr/bin/env python3
"""
Launcher script for the Todo App
"""

from todo_app import main
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application

if __name__ == "__main__":
    main()
