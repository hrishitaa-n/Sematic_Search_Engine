import psycopg2
conn = psycopg2.connect(host="localhost", user="postgres", password="1234", dbname="postgres")
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE DATABASE semanticdb;")
print("Database created")
conn.close()