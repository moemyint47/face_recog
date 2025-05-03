import sqlite3

# Connect to the database
conn = sqlite3.connect('face_embeddings.db')
cursor = conn.cursor()

# Get all user-defined tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Delete all entries from each table
for table_name in tables:
    print(f"Clearing table: {table_name[0]}")
    cursor.execute(f"DELETE FROM {table_name[0]}")
    conn.commit()

conn.close()
print("All tables have been emptied.")
