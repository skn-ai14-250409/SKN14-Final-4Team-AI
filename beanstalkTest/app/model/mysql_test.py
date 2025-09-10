# model/mysql_test.py

import mysql.connector
from mysql.connector import Error
from config import config


def list_admin():
    results = []
    with mysql.connector.connect(**config.MYSQL_DB_CONFIG) as conn:
        cur = conn.cursor()
        try:
            # SQL query 실행    
            # cur.execute("SELECT * FROM qna_question")
            # results = cur.fetchall()
            # Call stored procedure
            cur.callproc("SP_L_ADMIN")
            for result in cur.stored_results():
                results.append(result.fetchall())

        except Error as err:
            print(f"Error: {err}")
            results = False
        return results