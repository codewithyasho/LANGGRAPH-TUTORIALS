#!/usr/bin/env python3
"""
Test script for the Todo App
This script demonstrates the core functionality of the Todo application.
"""

import os
import sys
from todo_app import TodoApp

def test_todo_app():
    """Test the TodoApp functionality"""
    # Create a test instance with a separate file
    test_filename = "test_todos.json"
    app = TodoApp(test_filename)
    
    print("Testing Todo App functionality...")
    print("=" * 40)
    
    # Test adding todos
    print("1. Adding sample todos...")
    app.add_todo("Buy groceries", "Milk, eggs, bread", "high")
    app.add_todo("Walk the dog", "Evening walk", "medium")
    app.add_todo("Read a book", "Finish chapter 5", "low")
    
    # Test listing todos
    print("\n2. Listing todos...")
    app.list_todos()
    
    # Test completing a todo
    print("\n3. Completing todo #1...")
    app.complete_todo(1)
    
    # Test editing a todo
    print("\n4. Editing todo #2...")
    app.edit_todo(2, description="Morning walk around the park")
    
    # Test listing todos again to see changes
    print("\n5. Updated todo list...")
    app.list_todos()
    
    # Test deleting a todo
    print("\n6. Deleting todo #3...")
    app.delete_todo(3)
    
    # Final list
    print("\n7. Final todo list...")
    app.list_todos()
    
    # Clean up test file
    if os.path.exists(test_filename):
        os.remove(test_filename)
        print(f"\nCleaned up test file: {test_filename}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_todo_app()