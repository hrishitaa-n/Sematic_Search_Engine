import sys
sys.path.append('src')

from database import engine, Base
from models import Document

Base.metadata.create_all(bind=engine)
print("Table created successfully")