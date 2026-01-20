import uuid

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base


class Symptoms(Base):
    __tablename__ = "symptoms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(String(1024), nullable=True)
    severity_level = Column(Integer, nullable=True)  # e.g., 1-10 scale
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)