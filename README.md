# Applications of ML in Finance

## Description

## Installation and Set-Up
- Install dependencies
  - Run: `pip install -r requirements.txt`
- Configuring MySQL
  - Install [MySQL Workbench](https://dev.mysql.com/downloads/workbench/)
  - In MySQL Workbench create a new database and user
  - Run financial_db.sql script to create the tables
  - Create a file called passwords.py (because I didn't want to publish my passwords online)
    - This is the entire script:
      ```python
      # insert_symbols.py
      PASSWORD = "{your_MySQL_password_here}"
      API_KEY = "{your__Quandl_api_key_here}"
      ```
    - Import these variables with `from passwords.py import PASSWORD, API_KEY` whenever necessary. I've already done that for you for all included scripts.

## To Do's
- Configure postgres database
- Set up requirements.txt
- Clarify project scope
