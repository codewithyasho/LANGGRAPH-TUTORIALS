# Python Todo App

A simple command-line Todo application built with Python that allows you to manage your daily tasks.

## Features

- Add new todos with title, description, and priority
- List all todos with their status and priority
- Mark todos as completed
- Edit existing todos
- Delete todos
- Data persistence using JSON file storage

## How to Run

1. Make sure you have Python installed on your system
2. Navigate to the `python_todo_app` directory
3. Run the application:
   ```
   python todo_app.py
   ```

## Usage

When you run the application, you'll see a menu with the following options:

1. **Add Todo** - Create a new todo item with title, description, and priority
2. **List Todos** - Display all todos with their current status
3. **Complete Todo** - Mark a todo as completed
4. **Edit Todo** - Modify an existing todo's details
5. **Delete Todo** - Remove a todo from the list
6. **Exit** - Close the application

## Data Storage

Todos are automatically saved to a file named `todos.json` in the same directory. This file is created automatically when you add your first todo.

## Priority Levels

- High: 🔴
- Medium: 🟡
- Low: 🟢

## Status Indicators

- ○ - Incomplete
- ✓ - Completed

## Requirements

- Python 3.x