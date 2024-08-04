#!/usr/bin/env python3

# run this file in terminal for further memory optimization.
# python tools/fetch_dataset.py

import os
import pandas as pd
import mysql.connector
from typing import Dict, Optional
from dotenv import load_dotenv
from mysql.connector import MySQLConnection, Error


def load_env_vars() -> Dict[str, str]:
    load_dotenv()
    return {
        'db_host': os.getenv('DB_HOST'),
        'db_user': os.getenv('DB_USER'),
        'db_password': os.getenv('DB_PASSWORD'),
        'db_name': os.getenv('DB_NAME'),
        'db_table': os.getenv('DB_TABLE')
    }


def connect_to_database(config: Dict) -> Optional[MySQLConnection]:
    try:
        connection = mysql.connector.connect(
            host=config['db_host'],
            user=config['db_user'],
            password=config['db_password'],
            database=config['db_name']
        )
        if connection.is_connected():
            print('Database connection established successfully.')
            return connection
    except Error as e:
        print(f'Error connecting to database: {e}')
        return None


def fetch(connection, table_name) -> pd.DataFrame:
    query = f'SELECT * FROM {table_name}'
    try:
        df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        print(f'Error when executing a query: {e}')
        return pd.DataFrame()


def save_to_csv(df, filename):
    try:
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f'[ + ] {filename}')
    except Exception as e:
        print(f'Error saving CSV file: {e}')


def main():
    config = load_env_vars()
    connection = connect_to_database(config)

    if connection:
        df = fetch(connection, config['db_table'])
        save_to_csv(df, '../raw_dataset.csv')
        connection.close()


if __name__ == '__main__':
    main()
