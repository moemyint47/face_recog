import sqlite3

# Connect to the database
conn = sqlite3.connect('face_embeddings.db')
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# For each table, list all entries
for table_name in tables:
    print(f"\nContents of table '{table_name[0]}':")
    cursor.execute(f"SELECT * FROM {table_name[0]}")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

conn.close()
