import sqlite3

conn = sqlite3.connect("FaceDatabase.db")
cur = conn.cursor()

create_table_query = '''
    CREATE TABLE IF NOT EXISTS users (
    ID TEXT PRIMARY KEY NOT NULL,
    Name TEXT NOT NULL
)
'''
cur.execute(create_table_query)

conn.commit()
conn.close()