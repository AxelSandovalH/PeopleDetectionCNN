import sqlite3

# Conectar a la base de datos (se crea si no existe)
conn = sqlite3.connect('people_counter.db')
cursor = conn.cursor()

# Crear la tabla de eventos si no existe
cursor.execute('''
CREATE TABLE IF NOT EXISTS eventos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    direccion TEXT CHECK(direccion IN ('IN', 'OUT')) NOT NULL
)
''')

conn.commit()
conn.close()

print("Base de datos 'people_counter.db' y tabla 'eventos' creada correctamente.")
