from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.db import Base

engine = create_engine("sqlite:///cv_auto_trainer.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
