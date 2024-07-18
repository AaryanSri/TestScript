
import os

# Base directory of the project (one level above the current file)
BASE_DIR = os.path.join(os.path.dirname(__file__), os.pardir)

# File paths
COMPANY_HASHES_MAP_FILE = os.path.join(BASE_DIR, 'Stocks_data/company_hashes_map.txt')
LAST_UPDATED_FILE = os.path.join(BASE_DIR, 'Stocks_data/last_updated.txt')

# URLs
YAHOO_FINANCE_LOOKUP_URL = "https://finance.yahoo.com/lookup/"