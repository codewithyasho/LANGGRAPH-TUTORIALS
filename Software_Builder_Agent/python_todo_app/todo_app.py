import json
import os
from datetime import datetime

class TodoApp:
    def __init__(self, filename="todos.json"):
        self.filename = filename
        self.todos = []
        self.load_todos()
    
    def load_todos(self):
        """Load todos from file"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as file:
                    self.todos = json.load(file)
            except json.JSONDecodeError:
                self.todos = []
        else:
            self.todos = []
    
    def save_todos(self):
        """Save todos to file"""
        with open(self.filename, 'w') as file:
            json.dump(self.todos, file, indent=2)
    
    def add_todo(self, title, description="", priority="medium"):
        """Add a new todo item"""
        todo = {
            "id": len(self.todos) + 1,
            "title": title,
            "description": description,
            "priority": priority,
            "completed": False,
            "created_at": datetime.now().isoformat()
        }
        self.todos.append(todo)
        self.save_todos()
        print(f"Todo '{title}' added successfully!")
    
    def list_todos(self):
        """List all todos"""
        if not self.todos:
            print("No todos found.")
            return
        
        print("\n===== TODO LIST =====")
        for todo in self.todos:
            status = "✓" if todo["completed"] else "○"
            priority_symbol = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(todo["priority"], "⚪")
            print(f"{status} {priority_symbol} [{todo['id']}] {todo['title']}")
            if todo["description"]:
                print(f"   Description: {todo['description']}")
            print()
    
    def complete_todo(self, todo_id):
        """Mark a todo as completed"""
        for todo in self.todos:
            if todo["id"] == todo_id:
                todo["completed"] = True
                self.save_todos()
                print(f"Todo '{todo['title']}' marked as completed!")
                return
        print(f"Todo with ID {todo_id} not found.")
    
    def delete_todo(self, todo_id):
        """Delete a todo"""
        for i, todo in enumerate(self.todos):
            if todo["id"] == todo_id:
                deleted_title = todo["title"]
                del self.todos[i]
                # Update IDs to keep them sequential
                for j, remaining_todo in enumerate(self.todos):
                    remaining_todo["id"] = j + 1
                self.save_todos()
                print(f"Todo '{deleted_title}' deleted successfully!")
                return
        print(f"Todo with ID {todo_id} not found.")
    
    def edit_todo(self, todo_id, title=None, description=None, priority=None):
        """Edit a todo"""
        for todo in self.todos:
            if todo["id"] == todo_id:
                if title is not None:
                    todo["title"] = title
                if description is not None:
                    todo["description"] = description
                if priority is not None:
                    todo["priority"] = priority
                self.save_todos()
                print(f"Todo '{todo['title']}' updated successfully!")
                return
        print(f"Todo with ID {todo_id} not found.")

def main():
    app = TodoApp()
    
    while True:
        print("\n===== TODO APP MENU =====")
        print("1. Add Todo")
        print("2. List Todos")
        print("3. Complete Todo")
        print("4. Edit Todo")
        print("5. Delete Todo")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            title = input("Enter todo title: ")
            description = input("Enter todo description (optional): ")
            priority = input("Enter priority (high/medium/low) [medium]: ").lower() or "medium"
            if priority not in ["high", "medium", "low"]:
                priority = "medium"
            app.add_todo(title, description, priority)
        
        elif choice == "2":
            app.list_todos()
        
        elif choice == "3":
            app.list_todos()
            try:
                todo_id = int(input("Enter todo ID to mark as completed: "))
                app.complete_todo(todo_id)
            except ValueError:
                print("Invalid ID. Please enter a number.")
        
        elif choice == "4":
            app.list_todos()
            try:
                todo_id = int(input("Enter todo ID to edit: "))
                title = input("Enter new title (leave blank to keep current): ") or None
                description = input("Enter new description (leave blank to keep current): ") or None
                priority_input = input("Enter new priority (high/medium/low) (leave blank to keep current): ").lower()
                priority = priority_input if priority_input in ["high", "medium", "low"] else None
                app.edit_todo(todo_id, title, description, priority)
            except ValueError:
                print("Invalid ID. Please enter a number.")
        
        elif choice == "5":
            app.list_todos()
            try:
                todo_id = int(input("Enter todo ID to delete: "))
                confirm = input(f"Are you sure you want to delete todo #{todo_id}? (y/N): ")
                if confirm.lower() == 'y':
                    app.delete_todo(todo_id)
            except ValueError:
                print("Invalid ID. Please enter a number.")
        
        elif choice == "6":
            print("Thank you for using Todo App!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()