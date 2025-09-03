# model/mysql_test.py

import mysql.connector
from mysql.connector import Error
from config import config

def list_admin():
    with mysql.connector.connect(**config.MYSQL_DB_CONFIG) as conn:
        cur = conn.cursor()
        try:
            cur.execute("SELECT * FROM qna_question")
            results = cur.fetchall()
        except Error as err:
            print(f"Error: {err}")
            results = False
        return results

